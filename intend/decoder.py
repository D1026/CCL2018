import pandas as pd
import numpy as np

action_dict, target_dict, key_dict, value_dict = pd.read_pickle("yyzz_dict.pkl")
(text_test, positive_test, fake_act, fake_tar, fake_key, fake_val) = pd.read_pickle('fake_train_xpatkv.pkl')
act = fake_act[:10000]
tar = fake_tar[:10000]
key = fake_key[:10000]
val = fake_val[:10000]

# ----
# print(predict_action[:2])
# exit(888)


# def softmaxToOnehont(y):
#     shape = y.shape
#     y_ = np.zeros(shape=shape)
#     for i in range(shape[0]):
#         index = np.argmax(y[i])
#         y_[i][index] = 1
#     return y_
#
#
# action = softmaxToOnehont(predict_action)
# target = softmaxToOnehont(predict_target)
# key = softmaxToOnehont(predict_key)
# value = softmaxToOnehont(predict_value)


def onehotToClass(y, y_dict):
    y_list = []
    shape = y.shape
    for i in range(shape[0]):
        for k in y_dict.keys():
            # print(y[i])
            # print(y_dict[k])
            if (y[i] == y_dict[k]).all():
                y_list.append(k)
    return y_list

# ----------------- for xgb ----------------

# ------------------- end --------------------

action_list = onehotToClass(act, action_dict)
target_list = onehotToClass(tar, target_dict)
key_list = onehotToClass(key, key_dict)
value_list = onehotToClass(val, value_dict)

import json
with open('intent_data.recharge.testB.0925', 'r', encoding='UTF-8') as f:
    lines = f.readlines()

result = []
count = 0
for i in range(len(lines)):
    sample_dict = {"sentence": None, "intents": [{"action": {"value": ""}, "target": {"value": ""}}],
                   "slots": [{"key": "", "value": ""}]}
    # sample_json = json.dumps(sample_dict)
    id = lines[i].split('\t')[0]
    sentence = json.loads(lines[i].split('\t')[1])["sentence"]
    sample_dict["sentence"] = sentence

    if action_list[i] != "10086":
        sample_dict["intents"][0]["action"]["value"] = action_list[i]

    if target_list[i] != "10086":
        sample_dict["intents"][0]["target"]["value"] = target_list[i]

    if key_list[i] != "10010":
        sample_dict["slots"][0]["key"] = key_list[i]

    if value_list[i] != "10010":
        sample_dict["slots"][0]["value"] = value_list[i]

    if sample_dict["intents"][0]["action"]["value"] == "" and sample_dict["intents"][0]["target"]["value"] == "":
        sample_dict["intents"] = []

    if sample_dict["slots"][0]["key"] == "" and sample_dict["slots"][0]["value"] == "":
        sample_dict["slots"] = []

    sample = id + "\t" + json.dumps(sample_dict, ensure_ascii=False) + "\n"
    result.append(sample)
print(result)

with open("./result_0930E.txt", mode="w", encoding="utf-8") as f:
    f.writelines(result)
f.close()