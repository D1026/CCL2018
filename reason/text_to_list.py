import jieba
from jieba import posseg
import pickle
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

jieba.enable_parallel(4)


def text_to_list(pkl_name, x_train):
    train_dialogues = []
    train_dialogues_pos = []
    count = 0
    for x in x_train:
        train_s = []
        train_s_pos = []
        count += 1
        if count % 100 == 0:
            print(count)
        for s in x:
            words = []
            pos_seg = []
            line = posseg.lcut(s)
            line1 = jieba.lcut(s)
            # 删除信息量几乎为0的词
            # 考虑用 TFIDF 和 CHI 选词
            for i in line:
                words.append(i.word)
                pos_seg.append(i.flag)
            if len(words) != len(pos_seg):
                print("error")

            if len(words) > 0:
                train_s.append(words)
                train_s_pos.append(pos_seg)
                if len(words) != len(pos_seg):
                    print('error')
                    exit(-1)
        train_dialogues.append(train_s)
        train_dialogues_pos.append(train_s_pos)

    with open(pkl_name, 'wb') as f:
        pickle.dump((train_dialogues, train_dialogues_pos), f)


# 训练集
# /home/speech/voicefile/input/591_fujian_wf_5000_wf_01/24911576_1707127.V3	投诉（含抱怨）	业务使用问题
# 2	嗯嗯
# 1	下午好请说
# 2	你好我我这个宽带上不了网了去两个月也办不了到底是怎么回事啊

with open('callreason.train.fj_and_sh.2w', 'r', encoding='UTF-8') as train_txt:
    content = train_txt.read()
call_list = content.split('\n\n')
for i in range(2):
    print(call_list[i])
x_train = []
y1 = []
y2 = []
# 遍历每个来电数据，提取
for ele in call_list:
    if ele == '':
        continue
    sents = ele.split('\n')
    y_str = sents[0].split('\t')[1:]  # 两个元素或一个 一级分类 二级分类
    # y_str = ''.join(y_str)
    # print(y_str)
    # ['投诉（含抱怨）', '业务使用问题']
    try:
        y1.append(y_str[0])
        y2.append(y_str[1])
    except IndexError as e:
        print(IndexError)
        y_str = sents[1].split('\t')[1:]
        y1.append(y_str[0])
        y2.append(y_str[1])

    x_str = []  # 多条对话
    for i in sents[1:]:
        x_str.append(i.replace('\t', " "))
    x_train.append(x_str)

# train_data amount 20000
print(x_train[0])

# ---------------------------------------------------------------------------------------
# encoded_Y1, y1_categorical, encoded_Y2, y2_categorical
encoder = LabelEncoder()

encoded_Y1 = encoder.fit_transform(y1)
# convert integers to dummy variables (one hot encoding)
y1_categorical = np_utils.to_categorical(encoded_Y1)  # for LSTM

# make a yy_dict to recovery the prediction
y1_dict = {}
for i in encoder.classes_:
    y1_dict[i] = None

count = 0
for i in range(len(y1)):
    for k in y1_dict.keys():
        if y1[i] == k:
            y1_dict[k] = y1_categorical[i]
    count = count + 1
    if None not in y1_dict:
        continue
print(count)
print(y1_dict)

encoded_Y2 = encoder.fit_transform(y2)
y2_categorical = np_utils.to_categorical(encoded_Y2)

encoded_Y2 = encoder.fit_transform(y2)
y2_categorical = np_utils.to_categorical(encoded_Y2)
print(y2)

y2_dict = {}
for i in encoder.classes_:
    y2_dict[i] = None

count = 0
for i in range(len(y2)):
    for k in y2_dict.keys():
        if y2[i] == k:
            y2_dict[k] = y2_categorical[i]
    count = count + 1
    if None not in y2_dict:
        continue
print(count)
print(y2_dict)
# ---------------------------------------------------------------------------------------

# 分词
jieba.suggest_freq('兆', tune=True)
jieba.suggest_freq('块', tune=True)
jieba.suggest_freq('流量', tune=True)

# train_dialogues_and_pos.pkl


# 测试集
with open('callreason.testB_5K', 'r', encoding='UTF-8') as train_txt:
    content = train_txt.read()
call_list = content.split('\n\n')

x_test = []
ID = []
for ele in call_list:
    if ele == '':
        continue
    sents = ele.split('\n')
    # y_str = sents[0].split('\t')[1:]    # 两个元素或一个 一级分类 二级分类
    ID.append(sents[0])

    x_str = []  # 多条对话
    for i in sents[1:]:
        x_str.append(i.replace('\t', " "))
    x_test.append(x_str)

text_to_list('train_dialogues_and_pos.pkl', x_train)
text_to_list('testB_dialogues_and_pos.pkl', x_test)

with open('ID.pkl', 'wb') as f:
    pickle.dump(ID, f)
with open('4y.pkl', 'wb') as f:
    pickle.dump((encoded_Y1, y1_categorical, encoded_Y2, y2_categorical), f)

# ------------------------------------------------------------------------------------
with open('yy_dict.pkl', 'wb') as f:
    pickle.dump((y1_dict, y2_dict), f)

