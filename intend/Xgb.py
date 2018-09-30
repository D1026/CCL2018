import pickle
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack, vstack
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import json
from sklearn.ensemble import RandomForestClassifier
from jieba import posseg
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# -----------------------------------
# (act_x, tar_x, key_x, val_x) = pd.read_pickle('atkv0927_for_xgbTrain.pkl')  # 训练数据 x
# (act_pre, tar_pre, key_pre, val_pre) = pd.read_pickle('atkv0926.pkl')   # 待预测 x
# act_pre = act_pre[:10000]
# tar_pre = tar_pre[:10000]
# key_pre = key_pre[:10000]
# val_pre = val_pre[:10000]
(tfidf_train, tfidf_test) = pd.read_pickle('tfidf_train_test.pkl')  # 训练数据 x
(y1prob, y2prob, z1prob, z2prob) = pd.read_pickle('testB0928_prob.pkl')
# (y1, y2, z1, z2) = pd.read_pickle('encoded_y1y2z1z2.pkl')   # 训练y
# ----------------------------------
# 概率特征
(act_prob_pry, tar_prob_pry, key_prob_pry, val_prob_pry, act_prob_f, tar_prob_f, key_prob_f, val_prob_f) = pd.read_pickle('train1.8_prob.pkl')
(act_pred, tar_pred, key_pred, val_pred) = pd.read_pickle('test0.2_prob.pkl')
# x, p, tfidf, 特征 y1 y2 z1 z2 标签
(x_pry, p_pry, tfidf_pry, act_pry, tar_pry, key_pry, val_pry) = pd.read_pickle('shuffled_pry.pkl')
data01234 = pd.read_pickle('data01234.pkl')
(x_f_train, x_f_test, p_f_train, p_f_test, tfidf_f_train, tfidf_f_test, act_f_train, act_f_test, tar_f_train, tar_f_test, key_f_train, key_f_test, val_f_train, val_f_test) \
    = data01234[0]

def onehot2index(data):
    labels = []
    for i in data:
        labels.append(np.argmax(i))
    return labels


tfidf_train18 = vstack([tfidf_pry, tfidf_f_train], format='csr')
act_prob_train18 = np.concatenate([act_prob_pry, act_prob_f], axis=0)
tar_prob_train18 = np.concatenate([tar_prob_pry, tar_prob_f], axis=0)
key_prob_train18 = np.concatenate([key_prob_pry, key_prob_f], axis=0)
val_prob_train18 = np.concatenate([val_prob_pry, val_prob_f], axis=0)

act_label_train18 = np.concatenate([act_pry, act_f_train], axis=0)
tar_label_train18 = np.concatenate([tar_pry, tar_f_train], axis=0)
print(type(key_pry))
key_label_train18 = np.concatenate([key_pry, key_f_train], axis=0)
print(key_label_train18.shape)

val_label_train18 = np.concatenate([val_pry, val_f_train], axis=0)

y1_train = onehot2index(act_label_train18)
y2_train = onehot2index(tar_label_train18)
z1_train = onehot2index(key_label_train18)
print(set(z1_train))

z2_train = onehot2index(val_label_train18)


tfidf_test02 = tfidf_f_test
act_prob_test02 = act_pred
tar_prob_test02 = tar_pred
key_prob_test02 = key_pred
val_prob_test02 = val_pred

act_test02 = act_f_test
y1_test = onehot2index(act_test02)
y2_test = onehot2index(tar_f_test)
z1_test = onehot2index(key_f_test)
z2_test = onehot2index(val_f_test)


# X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, y1, test_size=0.33333, random_state=42)
#print('type(X_train)', type(X_train))   # <class 'scipy.sparse.csr.csr_matrix'>


# -------------------- for key ---------------------
sel = SelectKBest(chi2, k=50000)
tfidf_train_for_y1 = sel.fit_transform(tfidf_train18, y2_train)
tfidf_test_for_y1 = sel.transform(tfidf_test02)
tfidf_pred_for_y1 = sel.transform(tfidf_test)

X_train = hstack((tfidf_train_for_y1, act_prob_train18, tar_prob_train18), format='csr')
X_test = hstack((tfidf_test_for_y1, act_prob_test02, tar_prob_test02), format='csr')
X_pred = hstack((tfidf_pred_for_y1, y1prob, y2prob), format='csr')
# -------------------------------------
# X_predict = hstack((tfidf_test_for_y1, act_pre, tar_pre), format='csr')

# X_train, X_test, y_train, y_test = train_test_split(X_train, z2, test_size=0.33333, random_state=42)

train_data = X_train
test_data = X_test
# ---------------
# train_data = vstack((X_train, X_test), format='csr')
# train_label = z2_train + z2_test
# xgb_train = xgb.DMatrix(train_data, label=train_label)
# xgb_pred = xgb.DMatrix(X_pred)
# --------------------
xgb_train = xgb.DMatrix(train_data, label=y2_train)
xgb_test = xgb.DMatrix(test_data, label=y2_test)
xgb_pred = xgb.DMatrix(X_pred)
# ---------------
# xgb_pre = xgb.DMatrix(X_predict)
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',  # 多分类的问题
    # 'objective': 'binary:logistic',
    'num_class': 40,    #142,   # 7,  # 40,    # 13,  # 类别数，与 multisoftmax 并用
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 16,  # 构建树的深度，越大越容易过拟合
    'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.7,  # 生成树时进行的列采样
    'min_child_weight': 1,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.05,  # 如同学习率
    'seed': 1000,

    'nthread': 6,  # cpu 线程数
    'eval_metric': 'merror'
}

num_rounds = 31  # 迭代次数model
watchlist = [(xgb_train, 'train'), (xgb_test, 'val')]
# watchlist = [(xgb_train, 'train')]
# 训练模型
model = xgb.train(params, xgb_train, num_rounds, watchlist, early_stopping_rounds=32)

# ------------
y2_predict = model.predict(xgb_pred)

with open("xgb_y2_predict0928.pkl", "wb") as f:
    pickle.dump(y2_predict, f)
exit(1549)
# --------------------------------------------------------------

sel = SelectKBest(chi2, k=50000)
X_train = sel.fit_transform(X_train, y_train)
X_test = sel.transform(X_test)

# 稀疏矩阵拼接 vstack hstack
# 稀疏矩阵生成xgboost的dmatrix的时候,train 和 test数据集的非空行数不一致导致特征数不对
# 如果被迫还原成完整的非空数据集，使用6W数据集5000特征时候，内存溢出 所以强制增加一行非空特征
v = np.ones((X_train.shape[0], 1))
X_train = hstack((X_train, v), format='csr')
v = np.ones((X_test.shape[0], 1))
X_test = hstack((X_test, v), format='csr')
print(v.shape)
print(X_train.shape)
print(X_test.shape)

train_data = X_train
test_data = X_test
xgb_train = xgb.DMatrix(train_data, label=y_train)
xgb_test = xgb.DMatrix(test_data, label=y_test)
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',  # 多分类的问题
    # 'objective': 'binary:logistic',
    'num_class': 13,  # 类别数，与 multisoftmax 并用
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 16,  # 构建树的深度，越大越容易过拟合
    'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.7,  # 生成树时进行的列采样
    'min_child_weight': 1,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.05,  # 如同学习率
    'seed': 1000,

    'nthread': 6,  # cpu 线程数
    'eval_metric': 'merror'
}

num_rounds = 10000  # 迭代次数model
watchlist = [(xgb_train, 'train'), (xgb_test, 'val')]
# 训练模型
model = xgb.train(params, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)

exit(1234567)
# -------------------------------------------------------------