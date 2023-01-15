<!-- ## 训练结果
| 词向量 | 文本输入 | 模型 | 解码模块 | 收敛轮数 | Dev acc |
| ---- | ---- | ---- |  ---- |  ---- |  ---- |  
| original(TA) | asr_1best | BiLSTM | original | ? | 71.3966 |
| original(TA) | manual_transcript | BiLSTM | original | 87 | 93.1844 |
| sgns.zhihu.word | manual_transcript | BiLSTM | original | 48 | 94.1899  |
| sgns.zhihu.bigram-char | manual_transcript | BiLSTM | original | 40 | 93.9665  |
| sgns.context.word-word.dynwin5.thr10.neg5.dim300.iter5 | manual_transcript | BiLSTM | original | 8 | 93.0726 | -->

# 训练

## 创建环境
```bash
conda create -n slu python=3.6
source activate slu
pip install -r requirements.txt
# 如果运行中报错需要安装其他包，请pip install <package> 进行安装
```

## 训练
```bash
#step1: 先进行数据增广，否则训练可能会报错
python utils/data_aug.py

#step2: 运行训练脚本
bash shell-scripts/main.sh
# 或者 python scripts/slu_bert.py --<arg> <value>
```
### 数据目录说明：
- `data`
  - `train.json`: 原始训练数据
  - `train_augment.json`： 数据增广后的数据
  - `...` 
- `data_cais`:
  - `train.json`： cais数据集的训练数据
  - `...`
- `data_ecdt`:
  - `train.json`： ecdt数据集的训练数据
  - `...`

### 代码说明：
- `utils/args.py`: 定义了所有涉及到的可选参数
- `utils/batch.py`: 将数据以批为单位转化为输入
- `utils/data_aug.py`: 数据增广
- `utils/example_bert.py`: 读取数据
- `utils/initialization.py`: 初始化系统设置，包括设置随机种子和显卡/CPU
- `utils/vocab.py`: 构建编码输入输出的词表
- `utils/word2vec.py`: 读取词向量

# 测试
```bash
python scripts/slu_bert.py --device <device> --testing
```