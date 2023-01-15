import operator
import torch
from transformers import BertTokenizerFast, BertForMaskedLM
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


tokenizer = BertTokenizerFast.from_pretrained("shibing624/macbert4csc-base-chinese")
model = BertForMaskedLM.from_pretrained("../../..//huggingface/macbert4csc-base-chinese")
model.to(device)

import json
contents = json.load(open('/mnt/lustre/sjtu/home/zsz01/codes/homework/SLU-project-CS4314/data/train.json'))

texts = [item['asr_1best'] for content in contents for item in content]

with torch.no_grad():
    outputs = model(**tokenizer(texts, padding=True, return_tensors='pt').to(device))

def get_errors(corrected_text, origin_text):
    sub_details = []
    for i, ori_char in enumerate(origin_text):
        if ori_char in [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤']:
            # add unk word
            corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
            continue
        if i >= len(corrected_text):
            continue
        if ori_char != corrected_text[i]:
            if ori_char.lower() == corrected_text[i]:
                # pass english upper char
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                continue
            sub_details.append((ori_char, corrected_text[i], i, i + 1))
    sub_details = sorted(sub_details, key=operator.itemgetter(2))
    return corrected_text, sub_details

result = []
for ids, text in zip(outputs.logits, texts):
    _text = tokenizer.decode(torch.argmax(ids, dim=-1), skip_special_tokens=True).replace(' ', '')
    corrected_text = _text[:len(text)]
    corrected_text, details = get_errors(corrected_text, text)
    # print(text, ' => ', corrected_text, details)
    # result.append((corrected_text, details))
    result.append(corrected_text)

with open('asr_correct.txt', 'w') as wf, open('asr_diff.txt', 'w') as diff_f:
    for idx in range(len(result)):
        if texts[idx] != result[idx]:
            print(texts[idx], "===>", result[idx], file=diff_f)
        print(texts[idx], "===>", result[idx], file=wf)
