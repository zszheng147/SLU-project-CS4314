#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from transformers import BertModel

class SLU_decode(nn.Module):
    def __init__(self,):
        super().__init__()

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        prob, loss, sep_loss, tag_loss = self.forward(batch) # bsz * seqlen * [BIO]
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist() # 预测的类型 [BIO]
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0: # 一组BI结束了
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        return predictions, labels, loss.cpu().item()

class SLUTaggingBERTMultiHead(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.transformer=BertModel.from_pretrained(config.model_name)
        self.sep_layer=TaggingFNNDecoder(config.hidden_size, 4, config.tag_pad_idx,num_layers=2)
        self.act_layer=TaggingFNNDecoder(config.hidden_size, config.num_acts, config.tag_pad_idx,num_layers=2)
        self.slot_layer=TaggingFNNDecoder(config.hidden_size, config.num_slots, config.tag_pad_idx,num_layers=2)
        # self.sep_embed=nn.Linear(4,config.hidden_size)
        
    def forward(self, batch):
        
        tag_ids = batch.tag_ids
        sep_tag_ids = batch.sep_tag_ids
        act_ids = batch.act_ids
        slot_ids = batch.slot_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths
        
        trans_output=self.transformer(input_ids)
        trans_hidden=trans_output["last_hidden_state"]

        seps_prob, sep_loss = self.sep_layer(trans_hidden, tag_mask, sep_tag_ids)
        acts_prob, act_loss = self.act_layer(trans_hidden, tag_mask, act_ids)
        slots_prob, slot_loss = self.slot_layer(trans_hidden, tag_mask, slot_ids)

        # sep_prob , sep_loss=self.sep_layer(trans_hidden, tag_mask,sep_tag_ids)
        # sep_embedded=self.sep_embed(sep_prob)
        # decoder_out=self.decoder(input_ids,encoder_hidden_states=trans_hidden+sep_embedded)

        loss = sep_loss + act_loss + slot_loss

        return (seps_prob,acts_prob,slots_prob), loss, sep_loss, act_loss+slot_loss

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        prob, loss, sep_loss, tag_loss = self.forward(batch) # bsz * seqlen * [BIO]
        seps_prob,acts_prob,slots_prob=prob
        predictions = []
        for i in range(batch_size):
            sep_pred = torch.argmax(seps_prob[i], dim=-1).cpu().tolist() # 预测的类型 [BIO]
            acts_pred = torch.argmax(acts_prob[i], dim=-1).cpu().tolist() # 预测的类型 [BIO]
            slots_pred= torch.argmax(slots_prob[i], dim=-1).cpu().tolist() # 预测的类型 [BIO]
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            sep_pred = sep_pred[:len(batch.utt[i])]
            acts_pred = acts_pred[:len(batch.utt[i])]
            slots_pred = slots_pred[:len(batch.utt[i])]
            pred=[label_vocab.get_tag_idx(sep_pred[j],acts_pred[j],slots_pred[j]) for j in range(len(batch.utt[i]))]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0: # 一组BI结束了
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        return predictions, labels, loss.cpu().item()

class SLUTaggingBERTCascaded(SLU_decode):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx,num_layers=1)
        self.transformer=BertModel.from_pretrained(config.model_name)
        self.sep_layer=TaggingFNNDecoder(config.hidden_size, 4, config.tag_pad_idx,num_layers=2)
        self.decoder=BertModel.from_pretrained(config.model_name,is_decoder=True,add_cross_attention=True)

        # self.sep_embed=nn.Linear(4,config.hidden_size)
        
    
    def forward(self, batch):
        tag_ids = batch.tag_ids
        sep_tag_ids=batch.sep_tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths
        
        trans_output=self.transformer(input_ids)
        trans_hidden=trans_output["last_hidden_state"]

        _ , sep_loss=self.sep_layer(trans_hidden, tag_mask,sep_tag_ids)
        decoder_out=self.decoder(input_ids,encoder_hidden_states=trans_hidden)
        
        # sep_prob , sep_loss=self.sep_layer(trans_hidden, tag_mask,sep_tag_ids)
        # sep_embedded=self.sep_embed(sep_prob)
        # decoder_out=self.decoder(input_ids,encoder_hidden_states=trans_hidden+sep_embedded)

        hidden=decoder_out["last_hidden_state"]
        tag_output,tag_loss = self.output_layer(hidden, tag_mask, tag_ids)

        loss=sep_loss+tag_loss

        return tag_output, loss, sep_loss, tag_loss

class SLUTaggingBERT(SLU_decode):
    def __init__(self, config):
        super(SLUTaggingBERT, self).__init__()
        self.config = config
        # self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx,num_layers=1)
        self.transformer=BertModel.from_pretrained(config.model_name)
    
    def forward(self, batch):
        tag_ids = batch.tag_ids
        sep_tag_ids=batch.sep_tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths
        
        trans_output=self.transformer(input_ids)
        hidden=trans_output["last_hidden_state"]

        # print(hidden.shape,output.shape,input_ids.shape)
        tag_output,tag_loss = self.output_layer(hidden, tag_mask, tag_ids)

        return tag_output, tag_loss, 0, 0

class TaggingFNNDecoder(nn.Module):
    # 线性层输出 算交叉熵

    # 如何利用slot

    def __init__(self, input_size, num_tags, pad_id, num_layers=1):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        if num_layers==1:
            self.output_layer = nn.Linear(input_size, num_tags)
        else:
            self.output_layer=nn.Sequential(nn.Linear(input_size, input_size),
                                              nn.Tanh(), nn.Linear(input_size, num_tags))
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        
        logits = self.output_layer(hiddens)
        # print(logits.shape,"logits",mask.shape,"mask",self.num_tags)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return prob, torch.tensor(0.)

class SLUTagging(SLU_decode):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths

        # embed = nn.Dropout(0.05)(self.fc(self.word_embed(input_ids)))
        embed = self.word_embed(input_ids) 
        # print(embed.shape,"embed shape")
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        hiddens = self.dropout_layer(rnn_out)
        print(hiddens.shape,"hiddens") #(32 47 512)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)
        # print(tag_output.shape,"tagout")
        return tag_output

