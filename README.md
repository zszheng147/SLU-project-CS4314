### 创建环境

    conda create -n slu python=3.6
    source activate slu
    pip install torch==1.7.1

### 运行
    
在根目录下运行

    python scripts/slu_baseline.py

### 代码说明

+ `utils/args.py`:定义了所有涉及到的可选参数，如需改动某一参数可以在运行的时候将命令修改成
        
        python scripts/slu_baseline.py --<arg> <value>
    其中，`<arg>`为要修改的参数名，`<value>`为修改后的值
+ `utils/initialization.py`:初始化系统设置，包括设置随机种子和显卡/CPU
+ `utils/vocab.py`:构建编码输入输出的词表
+ `utils/word2vec.py`:读取词向量
+ `utils/example.py`:读取数据
+ `utils/batch.py`:将数据以批为单位转化为输入
+ `model/slu_baseline_tagging.py`:baseline模型
+ `scripts/slu_baseline.py`:主程序脚本

### 有关预训练语言模型

本次代码中没有加入有关预训练语言模型的代码，如需使用预训练语言模型我们推荐使用下面几个预训练模型，若使用预训练语言模型，不要使用large级别的模型
+ Bert: https://huggingface.co/bert-base-chinese
+ Bert-WWM: https://huggingface.co/hfl/chinese-bert-wwm-ext
+ Roberta-WWM: https://huggingface.co/hfl/chinese-roberta-wwm-ext
+ MacBert: https://huggingface.co/hfl/chinese-macbert-base

### 推荐使用的工具库
+ transformers
  + 使用预训练语言模型的工具库: https://huggingface.co/
+ nltk
  + 强力的NLP工具库: https://www.nltk.org/
+ stanza
  + 强力的NLP工具库: https://stanfordnlp.github.io/stanza/
+ jieba
  + 中文分词工具: https://github.com/fxsjy/jieba

### 助教提供了几个思考和探索的方向:
+ 对话历史该如何使用（严禁使用未来的对话来预测当前的对话）
+ 如何解决输入中的噪音问题
+ 遇到没有见过的槽值该如何解决
+ 除了序列标注方法外还有没有其他建模方式
    + 使用分类器预测value的方法训练效果很差，大家避免在该方向上踩坑

### 助教提供的参考论文
+ [Supervised Sequence Labelling with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/preprint.pdf) 有点长，137页
+ [Slot Tagging for Task Oriented Spoken Language Understanding in Human-to-human Conversation Scenarios](https://aclanthology.org/K19-1071.pdf)
+ [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)
+ [An End-to-end Approach for Handling Unknown Slot Values in Dialogue State Tracking](https://arxiv.org/pdf/1805.01555.pdf)
+ [Get To The Point: Summarization with Pointer-Generator Network](https://arxiv.org/pdf/1704.04368.pdf)
+ [Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems](https://arxiv.org/pdf/1905.08743.pdf)
