import json
import jieba.posseg as ps
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# -----训练数据------
with open('intent_data.train.0903', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
train_samples = []
train_ids = []
for i in lines:
    train_ids.append(i.split('\t')[0])
    train_samples.append(json.loads(i.split('\t')[1]))

# ------预测数据------
# with open('intent_data.test_B.5k.raw.0827', 'r', encoding='UTF-8') as f:
#     lines = f.readlines()
# testB_samples = []
# testB_ids = []
# for i in lines:
#     testB_ids.append(i.split('\t')[0])
#     testB_samples.append(json.loads(i.split('\t')[1]))
# ---训练数据----
# 拆解 句子、意图、槽位三项
train_sts = []
train_ints = []
train_slos = []
for sp in train_samples:
    int = []
    slo = []
    train_sts.append(sp['sentence'])
    if len(sp['intents']) > 0:
        for it in sp['intents']:
            int.append(it['action']['value'] + '*' + it['target']['value'])
    if len(sp['slots']) > 0:
        for sl in sp['slots']:
            slo.append(sl['key'] + '*' + sl['value'])
    train_ints.append(int)
    train_slos.append(slo)
# test
print('sts数量： '+str(len(train_sts)))
print('ints长度： '+str(len(train_ints)))
print('slos长度： '+str(len(train_slos)))

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(train_ints)
z = mlb.fit_transform(train_slos)
# test
print(mlb.classes_)
print(z[1081])
print(y[88])
print('y的维度：' + str(len(y[0])))     # 47
print('z的维度：' + str(len(z[0])))     # 194

import pickle
with open('multi_yz.pkl', 'wb') as f:
    pickle.dump((y, z), f)