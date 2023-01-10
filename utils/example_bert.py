import json
import jieba
import re
from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator
from transformers import BertTokenizer, BertModel

class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None, tokenizer_name=None, extend=False):
        cls.evaluator = Evaluator() # 评价
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path) #no use now
        cls.word2vec = Word2vecUtils(word2vec_path) # 词向量
        cls.label_vocab = LabelVocab(root,extend=extend) # ['B', 'I', 'O', '<pad>']
        cls.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    @classmethod
    def load_dataset(cls, data_path,data_path2=None):
        datas = json.load(open(data_path, 'r'))
        examples = []
        for data in datas:
            for utt in data:
                ex = cls(utt)
                examples.append(ex)
        if data_path2 is not None:
            datas = json.load(open(data_path2, 'r'))
            for data in datas:
                for utt in data:
                    ex = cls(utt)
                    examples.append(ex)
        return examples
    


    def __init__(self, ex: dict):
        super(Example, self).__init__()
        self.ex = ex
        
        # self.utt = ex['manual_transcript']
        self.utt = ex['asr_1best']
        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        
        self.tags = ['O'] * len(self.utt)
        self.sep_tag_id=[1] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
                self.sep_tag_id[bidx: bidx + len(value)] = [2] * len(value)
                self.sep_tag_id[bidx] = 3
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        
        # self.input_idx_ori = [Example.word_vocab[c] for c in self.utt]
        
        
        non_chinese = re.findall(r'[^\u4e00-\u9fff]', self.utt)
        # Insert a space before each non-Chinese character
        for c in non_chinese:
            self.utt = self.utt.replace(c, " " + c)
        
        self.input_idx = Example.tokenizer(self.utt)["input_ids"][1:-1]
        # print(self.utt,self.input_idx_ori,self.input_idx) #bert tokenizer will pad <sos> and <eos>
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]

