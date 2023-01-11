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
        
        
        # non_chinese = ("(side)","(unknown)","(robot)","(dialect)")
        # Insert a space before each non-Chinese character
        self.utt_ori=self.utt

        replace_dict = {"(side)":"未知话语两侧", "(unknown)":"未知未知未知的内容", "(robot)":"未知的机器内容", "(dialect)":"未知未知的方言内容", "(noise)":"未知的噪声内容"," ":"空","null":"空的内容"
                } #"ok":"ok "*2, "ktv":"ktv "*3, "hi":"hi "*2, "beyond":"beyond "*6

        # 对奇怪字符的处理
        for key, value in replace_dict.items():
            self.utt = self.utt.replace(key, value)

        self.input_idx = Example.tokenizer(self.utt)["input_ids"][1:-1]

        # 对英文分词的处理 （逐字符与逐单词不符）
        words = set(re.findall(r'[a-zA-Z]+', self.utt))
        for word in words:
            # print(word)
            word_token_ori=Example.tokenizer(word)["input_ids"][1:-1]
            word_token=[]
            while len(word_token) < len(word):
                word_token = word_token + word_token_ori #c重复当前token，补齐至与word一样长
            if len(word_token) > len(word):
                word_token=word_token[0:len(word)]
            i=0
            while i < (len(self.input_idx) - len(word_token_ori) + 1):
                if self.input_idx[i:i+len(word_token_ori)] == word_token_ori:
                    self.input_idx[i:i+len(word_token_ori)] = word_token
                    i+=len(word)-1
                i+=1
        assert len(self.utt_ori) == len(self.input_idx), f"Mismatch in length: {self.utt_ori} {self.input_idx}"
        
        # if self.utt_ori!=self.utt or words is not None:
        #     print(self.utt_ori,self.utt)

        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]

