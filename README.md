## 需要解决的问题：
- [中文词向量（词级别）](https://github.com/Embedding/Chinese-Word-Vectors/blob/master/README_zh.md)之[模型](https://jbox.sjtu.edu.cn/l/a19WDe) or 预训练语言模型（句子级别的表征）
- 模型架构的调整（biLSTM 换成 transformer）、BiLSTM参数优化（油水少）
- ASR结果优化模块
- 解码模块换成CTC, 强制对齐
- 文本数据加噪
- asr和人工标注的数据训练顺序

## 训练结果
| 词向量 | 文本输入 | 模型 | 解码模块 | 收敛轮数 | Dev acc |
| ---- | ---- | ---- |  ---- |  ---- |  ---- |  
| original(TA) | asr_1best | BiLSTM | original | ? | 71.3966 |
| original(TA) | manual_transcript | BiLSTM | original | 87 | 93.1844 |
| sgns.zhihu.word | manual_transcript | BiLSTM | original | 48 | 94.1899  |
| sgns.zhihu.bigram-char | manual_transcript | BiLSTM | original | 40 | 93.9665  |
| sgns.context.word-word.dynwin5.thr10.neg5.dim300.iter5 | manual_transcript | BiLSTM | original | 8 | 93.0726 |


### 有关预训练语言模型
本次代码中没有加入有关预训练语言模型的代码，如需使用预训练语言模型我们推荐使用下面几个预训练模型，若使用预训练语言模型，不要使用large级别的模型
+ Bert: https://huggingface.co/bert-base-chinese
+ Bert-WWM: https://huggingface.co/hfl/chinese-bert-wwm-ext
+ Roberta-WWM: https://huggingface.co/hfl/chinese-roberta-wwm-ext
+ MacBert: https://huggingface.co/hfl/chinese-macbert-base

### 推荐使用的工具库
+ transformers
  + 使用预训练语言模型的工具库: https://huggingface.co/
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

## 报告
[SJTU-overleaf](https://latex.sjtu.edu.cn/project/63a181c215dc190097c0b28d) 已邀请
