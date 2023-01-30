import json
import jieba
import re
from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator
from transformers import BertTokenizer, BertModel

class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator() # 评价
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path) # 词向量
        cls.label_vocab = LabelVocab(root) # ['B', 'I', 'O', '<pad>']
        cls.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-lert-base")

    @classmethod
    def load_dataset(cls, data_path):
        datas = json.load(open(data_path, 'r'))
        examples = []
        for data in datas:
            for utt in data:
                ex = cls(utt)
                examples.append(ex)
        return examples
    


    def __init__(self, ex: dict):
        super(Example, self).__init__()
        self.ex = ex
        
        self.utt = ex['manual_transcript']
        # self.utt = ex['asr_1best']
        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        # Use a regular expression to find all non-Chinese characters
        non_chinese = re.findall(r'[^\u4e00-\u9fff]', self.utt)

        # Insert a space before each non-Chinese character
        for c in non_chinese:
            self.utt = self.utt.replace(c, " " + c)
        self.input_idx_new = Example.tokenizer(self.utt)["input_ids"][1:-1]
        if len(self.input_idx)!=len(self.input_idx_new):
            print(len(self.input_idx),len(self.input_idx_new))
            print(self.utt,self.input_idx,self.input_idx_new)

        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
    #     utt_seg = list(jieba.cut(self.utt))
    #     self.tags = ['O'] * len(utt_seg)
    #     for slot in self.slot:
    #         value = self.slot[slot]

    #         bidx, eidx = self.search_list(utt_seg, value) #!
    #         if bidx != -1:
    #             self.tags[bidx: eidx] = [f'I-{slot}'] * (eidx - bidx)
    #             self.tags[bidx] = f'B-{slot}'

    #     self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
    #     self.input_idx = [Example.word_vocab[c] for c in utt_seg]
    #     l = Example.label_vocab
    #     self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]

    # def search_list(utt_seg, value):
    #     if value not in "".join(utt_seg):
    #         return -1, -1

    #     bidx, eidx = -1, -1
    #     length = len(value)
    #     for idx in reversed(range(length)):
    #         pass 
    










    # def __init__(self, ex: dict):
    #     super(Example, self).__init__()
    #     self.ex = ex

    #     # self.utt = ex['asr_1best']
    #     self.utt = ex['manual_transcript']
    #     self.slot = {}
    #     for label in ex['semantic']:
    #         act_slot = f'{label[0]}-{label[1]}'
    #         if len(label) == 3:
    #             self.slot[act_slot] = label[2]

    #     for v in self.slot.values():
    #         self.utt = self.utt.replace(v, f" {v} ")
        
    #     seg_list = list(jieba.cut(self.utt))
    #     while True:
    #         if ' ' in seg_list:
    #             seg_list.remove(' ')
    #         else:
    #             break

    #     self.tags = ['O'] * len(seg_list)
        
    #     for slot in self.slot:
    #         value = self.slot[slot]

    #         # bidx = self.utt.find(value)
    #         value_seg_list = list(jieba.cut(value, cut_all=False))
            
    #         bidx = 0
    #         while True:
    #             if bidx >= len(seg_list): break
    #             try:
    #                 bidx = seg_list.index(value_seg_list[0], bidx)
    #             except:
    #                 bidx = len(seg_list)
    #                 break

    #             prev_bidx = bidx
    #             for idx, seg in enumerate(value_seg_list[1:]):
    #                 try:
    #                     if seg not in seg_list[bidx+1+idx]:
    #                         bidx = bidx + 1
    #                         break
    #                 except:
    #                     bidx = len(seg_list)
    #             if bidx == prev_bidx:
    #                 break
            
    #         # assert bidx < len(seg_list), f"{value}"
    #         if bidx < len(seg_list):
    #             self.tags[bidx: bidx + len(value_seg_list)] = [f'I-{slot}'] * len(value_seg_list)
    #             self.tags[bidx] = f'B-{slot}'
    #     self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]

    #     #! use subword from jieba instead of word
    #     self.input_idx = [Example.word_vocab[c] for c in seg_list]
    #     l = Example.label_vocab
    #     self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]


