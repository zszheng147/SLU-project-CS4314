import json

# read train.json
with open("train_cais.json", "r") as f:
    data = json.load(f)

# initialize actions and slots lists
actions = []
slots = []

# iterate through data
for i in data:
    for j in i:
        if len(j["semantic"])!=0:
            for k in range(len(j["semantic"])):

                action = j["semantic"][k][0]
                slot = j["semantic"][k][1]
                
                # add unique actions and slots to lists
                if action not in actions:
                    actions.append(action)
                if slot not in slots:
                    slots.append(slot)
            
# write ontology.json
ontology = {"acts": actions, "slots": slots}
with open("ontology_cais.json", "w") as f:
    json.dump(ontology, f)
