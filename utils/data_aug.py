import random
import json

origin = json.load(open('../data/train.json'))
ontology = json.load(open('../data/ontology.json'))

## 这些slots的value都在poi_name文件中
poi_slots = ['poi名称', 'poi修饰', 'poi目标', '起点名称', '起点修饰', '起点目标', 
                '终点名称', '终点修饰', '终点目标', '途经点名称']
poi_values = [c.rstrip() for c in open('../data/lexicon/poi_name.txt')]


## 请求类型
request_values = ["附近", "定位", "近郊", "旁边", "周边", "就近", "最近"]
# acts, slots = ontology["acts"], ontology["slots"]

#  路线偏好
preferenece_values = ["最近", "高速优先", "走国道", "少走高速", "不走高速", "走高速",
            "上高速", "高速公路", "最快", "躲避拥堵"]


appendix = []

for idx in range(len(origin)):
    for utt_id in range(len(origin[idx])):

        manual_transcript = origin[idx][utt_id]['manual_transcript']
        semantic = origin[idx][utt_id]['semantic']
        for semantic_idx in range(len(semantic)):
            slot = semantic[semantic_idx][1]
            value = semantic[semantic_idx][2]
            if slot in poi_slots:
                for _ in range(20):
                    v = random.choice(poi_values)
                    tmp = origin[idx]
                    tmp[utt_id]["semantic"][semantic_idx][2] = v
                    tmp[utt_id]['manual_transcript'] = manual_transcript.replace(value, v)
                    appendix.append(tmp)
            
            if slot == "请求类型":
                for v in request_values:
                    tmp = origin[idx]
                    tmp[utt_id]["semantic"][semantic_idx][2] = v
                    tmp[utt_id]['manual_transcript'] = manual_transcript.replace(value, v)
                    appendix.append(tmp)

            if slot == "路线偏好":
                for v in preferenece_values:
                    tmp = origin[idx]
                    tmp[utt_id]["semantic"][semantic_idx][2] = v
                    tmp[utt_id]['manual_transcript'] = manual_transcript.replace(value, v)
                    appendix.append(tmp)

print(len(appendix))
augment = json.dumps(appendix, indent=4, ensure_ascii=False)

with open('../data/train_augment.json', 'w') as wf:
    print(augment, file=wf)