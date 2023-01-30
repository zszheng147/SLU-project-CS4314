import json
import random

# load the train.json file
with open("train_all.json", "r") as f:
    data = json.load(f)

# shuffle the data
random.shuffle(data)

# split the data into a training set (80%) and a dev set (20%)
split_index = int(len(data) * 0.85)
train_data = data[:split_index]
dev_data = data[split_index:]

# write the new data to train_new.json and dev.json
with open("train.json", "w", encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False)

with open("development.json", "w", encoding='utf-8') as f:
    json.dump(dev_data, f, ensure_ascii=False)

