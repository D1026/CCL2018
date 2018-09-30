import json
import pickle
from jieba import posseg
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

with open('intent_data.train.0903', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
samples = []
for i in lines:
    samples.append(json.loads(i.split('\t')[1]))

print('复赛训练样本数：', len(samples))    # 20000

# ----------- 加上初赛数据集----------
with open('intent.train_data.2w', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
for i in lines:
    samples.append(json.loads(i.split('\t')[1]))
print('训练样本数：', len(samples))    # 40053


# 拆解 句子、意图、槽位三项
sentences = []
intent_action = []
intent_target = []
slots_key = []
slots_value = []
intent_num = []
slots_num = []

mulLabelSamps = []

for sp in samples:
    int_action = []
    int_target = []
    slot_key = []
    slot_value = []

    sentences.append(sp['sentence'])
    if len(sp['intents']) > 0:
        for it in sp['intents']:
            int_action.append(it['action']['value'])
            int_target.append(it['target']['value'])
    # test
    if len(sp['intents']) > 1:
        print(sp)

    if len(sp['slots']) > 0:
        for sl in sp['slots']:
            slot_key.append(sl['key'])
            slot_value.append(sl['value'])
    # test
    if len(sp['slots']) > 1:
        print(sp)
    if len(sp['intents']) > 1 or len(sp['slots']) > 1:
        mulLabelSamps.append(sp)


    intent_action.append(int_action)
    intent_target.append(int_target)
    slots_key.append(slot_key)
    slots_value.append(slot_value)

    intent_num.append(len(sp['intents']))
    slots_num.append(len(sp['slots']))

with open('mulLabelSamps', 'w', encoding='UTF-8') as f:
    json.dump(mulLabelSamps, f)
# exit(666)

intentNum_categories = np_utils.to_categorical(intent_num)
slotsNum_categories = np_utils.to_categorical(slots_num)
print(intentNum_categories[:10])
print(slotsNum_categories[:10])
with open("intNum_sloNum_oneHot.pkl", "wb") as f:
    pickle.dump((intentNum_categories, slotsNum_categories), f)
print(len(intent_num))
print(intent_num[:10])
print(len(slots_num))
print(slots_num[:10])
a = intent_num.count(2)
b = slots_num.count(2) + slots_num.count(3)
print('2 intend count: ' + str(a))
print('2,3 slots count: ' + str(b))
print(1 - a/40002)
print(1 - b/40002)


# 意图动作和意图目标
intent_action1 = []
intent_action2 = []

intent_target1 = []
intent_target2 = []


for i in range(len(intent_action)):
    if len(intent_action[i]) == 0:
        intent_action1.append("")
        # intent_action2.append("")

    elif len(intent_action[i]) == 1:
        intent_action1.append(intent_action[i][0])
        # intent_action1.append("")

    elif len(intent_action[i]) == 2:
        intent_action1.append(intent_action[i][0])
        # intent_action1.append(intent_action[i][1])
    else:
        pass

for i in range(len(intent_target)):
    if len(intent_target[i]) == 0:
        intent_target1.append("")
        # intent_target2.append("")

    elif len(intent_target[i]) == 1:
        intent_target1.append(intent_target[i][0])
        # intent_target2.append("")

    elif len(intent_target[i]) == 2:
        intent_target1.append(intent_target[i][0])
        # intent_target2.append(intent_target[i][1])
    else:
        pass

# 槽位
slots_key1 = []
slots_key2 = []

slots_value1 = []
slots_value2 = []

for i in range(len(slots_key)):
    if len(slots_key[i]) == 0:
        slots_key1.append("")
        # intent_action2.append("")
    else:
        slots_key1.append(slots_key[i][0])
    # elif len(slots_key[i]) == 1:
    #     slots_key1.append(slots_key[i][0])
    #     # intent_action1.append("")
    #
    # elif len(slots_key[i]) == 2:
    #     slots_key1.append(slots_key[i][0])
    #     # intent_action1.append(intent_action[i][1])
    # else:
    #     pass

for i in range(len(slots_value)):
    if len(slots_value[i]) == 0:
        slots_value1.append("")
        # intent_target2.append("")
    else:
        slots_value1.append(slots_value[i][0])
# elif len(slots_value[i]) == 1:
    #     slots_value1.append(slots_value[i][0])
    #     # intent_target2.append("")
    #
    # elif len(intent_target[i]) == 2:
    #     slots_value1.append(slots_value[i][0])
    #     # intent_target2.append(intent_target[i][1])
    # else:
    #     pass

# intent_actions = []
# intent_targets = []

# intent_actions = intent_action1 + intent_action2
# intent_targets = intent_target1 + intent_target2

# 意图动作和意图目标编码
intent_actions = intent_action1
intent_targets = intent_target1

for i in range(len(intent_actions)):
    if intent_actions[i] == '':
        intent_actions[i] = "10086"

for i in range(len(intent_targets)):
    if intent_targets[i] == '':
        intent_targets[i] = "10086"


encoder = LabelEncoder()
encoded_actions = encoder.fit_transform(intent_actions)
actions_categories = np_utils.to_categorical(encoded_actions)
action_class = encoder.classes_
print('action类别数量：', len(action_class))    # 13

encoder = LabelEncoder()
encoded_targets = encoder.fit_transform(intent_targets)
targets_categories = np_utils.to_categorical(encoded_targets)
target_class = encoder.classes_
print('tar class:   ', target_class)

print('target类别数量：', len(target_class))    # 40


# 槽位编码
slots_keys = slots_key1
slots_values = slots_value1

for i in range(len(slots_keys)):
    if slots_keys[i] == '':
        slots_keys[i] = "10010"

for i in range(len(slots_values)):
    if slots_values[i] == '':
        slots_values[i] = "10010"


encoder = LabelEncoder()
encoded_keys = encoder.fit_transform(slots_keys)
keys_categories = np_utils.to_categorical(encoded_keys)
keys_class = encoder.classes_
print('key类别数量：', len(keys_class))    # 7

encoder = LabelEncoder()
encoded_values = encoder.fit_transform(slots_values)
values_categories = np_utils.to_categorical(encoded_values)
values_class = encoder.classes_
print('val类别数量：', len(values_class))    # 142

# 意图动作和意图目标真实标签和编码映射
action_dict = {}
for i in action_class:
    action_dict[i] = None

for i in range(len(intent_actions)):
    for k in action_dict.keys():
        if intent_actions[i] == k:
            action_dict[k] = actions_categories[i]

target_dict = {}
for i in target_class:
    target_dict[i] = None

for i in range(len(intent_targets)):
    for k in target_dict.keys():
        if intent_targets[i] == k:
            target_dict[k] = targets_categories[i]

# 意图动作和意图目标真实标签和编码映射
keys_dict = {}
for i in keys_class:
    keys_dict[i] = None

for i in range(len(slots_keys)):
    for k in keys_dict.keys():
        if slots_keys[i] == k:
            keys_dict[k] = keys_categories[i]

values_dict = {}
for i in values_class:
    values_dict[i] = None

for i in range(len(slots_values)):
    for k in values_dict.keys():
        if slots_values[i] == k:
            values_dict[k] = values_categories[i]
# -------  label: y,z 处理完毕 -------
train_text = []
train_positive = []
for i in sentences:
    words = []
    pos_seg = []
    line = posseg.lcut(i)
    for k in line:
        words.append(k.word)
        pos_seg.append(k.flag)
    if len(words) != len(pos_seg):
        print("error")
        break

    if len(words) > 0:
        train_text.append(words)
        train_positive.append(pos_seg)

# ------------------ 加载测试数据 --------------------
sentences_test = []
with open('intent_data.recharge.testB.0925', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
for i in lines:
    sentences_test.append(json.loads(i.split('\t')[1])["sentence"])
print('sentence num of test finalB: ', len(sentences_test))  # 10000

with open('intent_data.recharge.testA.0903', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
for i in lines:
    sentences_test.append(json.loads(i.split('\t')[1])["sentence"])
print('sentence num of test finalA: ', len(sentences_test))  # 5001

with open('intent_data.test_B.5k.raw.0827', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
for i in lines:
    sentences_test.append(json.loads(i.split('\t')[1])["sentence"])
print('sentence num of prytest B: ', len(sentences_test))  # 5001

with open('intent_data.testA_5K', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
for i in lines:
    sentences_test.append(json.loads(i.split('\t')[1])["sentence"])
print('sentence num of prytest A: ', len(sentences_test))  # 5000

with open('user_query.unlabeled.filter_2Words', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
lines = lines[1:100001]
for line in lines:
    sentences_test.append(line.strip())
print('all test sentence num:   ', len(sentences_test))     # 100000


test_text = []
test_positive = []
for i in sentences_test:
    words = []
    pos_seg = []
    line = posseg.lcut(i)
    for k in line:
        words.append(k.word)
        pos_seg.append(k.flag)
    if len(words) != len(pos_seg):
        print("error")
        break

    if len(words) > 0:
        test_text.append(words)
        test_positive.append(pos_seg)

segment_text = train_text + test_text   # 40053 +

# ------------------------ for Xgb --------------------
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

segment_text_xgb = train_text + test_text[:10000]
count_vec = CountVectorizer(ngram_range=(1, 4), token_pattern=r'\b\w+\b', min_df=1)
document_term_matrix = count_vec.fit_transform([" ".join(s) for s in segment_text_xgb])
# count_all = count_vec.transform([" ".join(s) for s in segment_text_xgb])
# count_train = count_vec.transform([" ".join(s) for s in train_text])
# count_test = count_vec.transform([" ".join(s) for s in test_text[:10000]])
vocabulary = count_vec.vocabulary_  # 得到词汇表
tf_idf_transformer = TfidfTransformer()
tf_idf_matrix = tf_idf_transformer.fit_transform(document_term_matrix)

print('tf_idf_matrix type:  ', type(tf_idf_matrix))

tfidf_train = tf_idf_matrix[:40053]
tfidf_test = tf_idf_matrix[40053:]
# tfidf_train = tf_idf_transformer.transform(count_train)
# tfidf_test = tf_idf_transformer.transform(count_test)

y1 = encoded_actions
y1_class = action_class
print(y1_class)
y2 = encoded_targets
y2_class = target_class
z1 = encoded_keys
z1_class = keys_class
z2 = encoded_values
z2_class = values_class

with open("tfidf_train_test.pkl", "wb") as f:
    pickle.dump((tfidf_train, tfidf_test), f)
with open("encoded_y1y2z1z2.pkl", "wb") as f:
    pickle.dump((y1, y2, z1, z2), f)
with open("encoded_y1y2z1z2_class.pkl", "wb") as f:
    pickle.dump((y1_class, y2_class, z1_class, z2_class), f)
# ------------------------ end -----------------------

tokenizer = Tokenizer(num_words=30000)
tokenizer.fit_on_texts(segment_text)
word_index = tokenizer.word_index
print(word_index)
text = tokenizer.texts_to_sequences(segment_text)
text_symbols = len(tokenizer.word_index) + 1
print(text_symbols)    # 16287
text = pad_sequences(text, truncating='pre')

print('-----', text.shape)  # (155053, 31)

# ----------------- word2vector -----------------
from gensim.models import Word2Vec
model = Word2Vec(segment_text, size=100, window=5, min_count=1, workers=4)
word_vectors = model.wv
print('word to vector:  ', word_vectors['流量'])
del model

# def seq2vectors(text):
#     textVec = []
#     for st in text:
#         stVec = []
#         for w in st:
#             stVec.append(word_vectors[w])
#         textVec.append(stVec)
embedding_matrix = np.zeros((len(word_index)+1, 100))
for word, index in word_index.items():
    embedding_matrix[index] = word_vectors[word]

with open("embedding_matrix.pkl", "wb") as f:
    pickle.dump(embedding_matrix, f)
# ----------------- end ------------------------

segment_positive = train_positive + test_positive
tokenizer = Tokenizer()
tokenizer.fit_on_texts(segment_positive)
positive = tokenizer.texts_to_sequences(segment_positive)
positive_symbols = len(tokenizer.word_index) + 1
print(positive_symbols)    # 54
positive = pad_sequences(positive, truncating='pre')

text_train = text[0:len(train_text)]
positive_train = positive[0:len(train_positive)]

text_test = text[len(train_text):]
positive_test = positive[len(train_positive):]
print(len(text_train))  # 40053
print(len(text_test))   # 125002

# text_test：[5001:] 为无标签数据
with open('xxxxyyzz.pkl', 'wb') as f:
    pickle.dump((text_train, positive_train, text_test, positive_test, actions_categories, targets_categories, keys_categories, values_categories), f)

with open("yyzz_dict.pkl", "wb") as f:
    pickle.dump((action_dict, target_dict, keys_dict, values_dict), f)


# -------------- ------
'''
    train:  20000 : 20053   -- 复赛训练 ：初赛训练---
    test:   10000 ： 5001 : 5001 : 5000 : 10 0000    -- 复赛测试B ：复赛测试A ： 初赛测试B ： 初赛A ： 无标签十万 ----
    '''