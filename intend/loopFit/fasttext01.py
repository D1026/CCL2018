import pandas as pd
import numpy as np
from numpy import random
import pickle
from scipy.sparse import vstack
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from intend.attenLayer import AttLayer
from sklearn.model_selection import train_test_split

np.random.seed(42)
label_num = [13, 40, 7, 142]
vocab_dim = 512
pos_dim = 32
vec_dim = 100
vocabulary_size = 16287
pos_size = 54

embedding_matrix = pd.read_pickle('embedding_matrix.pkl')


def build_input(input_dim, output_dim, shape):
    inputs = Input(shape=shape)
    x = Embedding(output_dim=output_dim, input_dim=input_dim + 1, mask_zero=False,
                    input_length=shape[0])(inputs)

    return inputs, x


def vec_input(shape):
    inputs = Input(shape=shape)
    x = Embedding(input_dim=16287, output_dim=100, weights=[embedding_matrix], mask_zero=False,
                  input_length=shape[0], trainable=False)(inputs)

    return inputs, x


def concat_output(x_1, x_2, x_3, vocab_dimension, pos_dimension, vec_dim):
    x = Concatenate()([x_1, x_2, x_3])
    x = Dropout(0.5)(x)
    x = AveragePooling1D(pool_size=1)(x)
    x = Reshape((-1, vocab_dimension + pos_dimension + vec_dim))(x)
    print(x.shape)
    x = Dropout(0.5)(x)
    x = Bidirectional(GRU(300, return_sequences=True))(x)
    x = AttLayer(300)(x)

    return x


(text_train, positive_train, text_test, positive_test, action, target, key, value) = pd.read_pickle('xxxxyyzz.pkl')
(tfidf_train, tfidf_test) = pd.read_pickle('tfidf_train_test.pkl')
print(len(action))

# -----
text_test = text_test[:10000]
positive_test = positive_test[:10000]

# --------切割初赛复赛数据集--------
x_final = text_train[:20000]
p_final = positive_train[:20000]
tfidf_final = tfidf_train[:20000]
act_final = action[:20000]
tar_final = target[:20000]
key_final = key[:20000]
val_final = value[:20000]

x_preliminary = text_train[20000:40053]
p_preliminary = positive_train[20000:40053]
tfidf_preliminary = tfidf_train[20000:40053]
act_preliminary = action[20000:40053]
tar_preliminary = target[20000:40053]
key_preliminary = key[20000:40053]
val_preliminary = value[20000:40053]

# ------------ shuffle  ---------
def shuffle(x, p, tfidf, y1, y2, z1, z2):
    if len(x) != len(y1):
        print('shuffle 数据与标签数量不等')
        exit(666)
    index = [i for i in range(len(x))]
    random.shuffle(index)
    shuffled_x = []
    shuffled_p = []
    shuffled_tfidf = tfidf[5]
    shuffled_y1 = []
    shuffled_y2 = []
    shuffled_z1 = []
    shuffled_z2 = []
    for i in index:
        shuffled_x.append(x[i])
        shuffled_p.append(p[i])
        shuffled_tfidf = vstack((shuffled_tfidf, tfidf[i]), format='csr')
        shuffled_y1.append(y1[i])
        shuffled_y2.append(y2[i])
        shuffled_z1.append(z1[i])
        shuffled_z2.append(z2[i])
    return shuffled_x, shuffled_p, shuffled_tfidf[1:], shuffled_y1, shuffled_y2, shuffled_z1, shuffled_z2


x_final, p_final, tfidf_final, act_final, tar_final, key_final, val_final = shuffle(x_final, p_final, tfidf_final, act_final, tar_final, key_final, val_final)
x_pry, p_pry, tfidf_pry, act_pry, tar_pry, key_pry, val_pry = shuffle(x_preliminary, p_preliminary, tfidf_preliminary, act_preliminary, tar_preliminary, key_preliminary, val_preliminary)

def split(x, i):
    x1 = x[:4000*i] + x[(i+1)*4000:]
    x2 = x[4000*i:4000*(i+1)]
    return x1, x2


def splitSparse(x, i):
    print(type(x))
    print(x.shape)
    xbf = x[:4000*i]
    xbh = x[(i+1)*4000:]
    x1 = vstack((xbf, xbh), format='csr')
    x2 = x[4000 * i:4000 * (i + 1)]
    return x1, x2


data01234 = []
for i in range(5):
    x_f_train, x_f_test = split(x_final, i)
    p_f_train, p_f_test = split(p_final, i)
    tfidf_f_train, tfidf_f_test = splitSparse(tfidf_final, i)
    act_f_train, act_f_test = split(act_final, i)
    tar_f_train, tar_f_test = split(tar_final, i)
    key_f_train, key_f_test = split(key_final, i)
    val_f_train, val_f_test = split(val_final, i)
    data01234.append((x_f_train, x_f_test, p_f_train, p_f_test, tfidf_f_train, tfidf_f_test, act_f_train, act_f_test, tar_f_train, tar_f_test, key_f_train, key_f_test, val_f_train, val_f_test))

with open("data01234.pkl", "wb") as f:
    pickle.dump(data01234, f)
with open("shuffled_pry.pkl", "wb") as f:
    pickle.dump((x_pry, p_pry, tfidf_pry, act_pry, tar_pry, key_pry, val_pry), f)
# --------------------for validation------------------
# X_train, X_test, P_train, P_test, action_train, action_test, target_train, target_test, key_train, key_test, value_train, value_test\
#     = train_test_split(x_final, p_final, act_final, tar_final, key_final, val_final, random_state=42, test_size=0.3)

# print('X_test : ', len(X_test))
# print('X_train type:  ', type(X_train))
# print('X_train[0] type:  ', type(X_train[0]))
# print('x_pry[0] type:  ', type(x_pry[0]))

# ----------------------------------
# X_train = np.concatenate([x_pry, X_train])
# P_train = np.concatenate([p_pry, P_train])
# action_train = np.concatenate([act_pry, action_train])
# target_train = np.concatenate([tar_pry, target_train])
# key_train = np.concatenate([key_pry, key_train])
# value_train = np.concatenate([val_pry, value_train])

# print('X_train 合并后：', len(X_train))
def getmodel(vocabulary_size = 16287, pos_size = 54, vocab_dim = 512, pos_dim = 32):
    inputs_a, x_a = build_input(vocabulary_size, vocab_dim, (31,))
    inputs_b, x_b = build_input(pos_size, pos_dim, (31,))
    inputs_c, x_c = vec_input((31,))

    x_1 = concat_output(x_a, x_b, x_c, vocab_dim, pos_dim, vec_dim)

    y1 = Dense(13, activation='softmax', name="action1")(x_1)
    y2 = Dense(40, activation='softmax', name="target1")(x_1)
    z1 = Dense(7, activation='softmax', name="key1")(x_1)
    z2 = Dense(142, activation='softmax', name="value1")(x_1)
    # ----------------- start  -----------------
    # y = Concatenate()([y1, y2])
    # z = Concatenate()([z1, z2])
    #
    # predictions_a = Dense(13, activation='softmax', name="action")(y)
    # predictions_b = Dense(40, activation='softmax', name="target")(y)
    # predictions_c = Dense(7, activation='softmax', name="key")(z)
    # predictions_d = Dense(142, activation='softmax', name="value")(z)
    # -------------- end ---------------------
    model = Model(inputs=[inputs_a, inputs_b, inputs_c],
                  outputs=[y1, y2, z1, z2])
    return model

model = getmodel()
model.save('model.h5')

# exit(666)
# --------- old code -----------
# inputs_a, x_a = build_input(vocabulary_size, vocab_dim, text_train[0].shape)
# inputs_b, x_b = build_input(pos_size, pos_dim, positive_train[0].shape)
#
#
# x_1 = concat_output(x_a, x_b, vocab_dim, pos_dim)
#
# predictions_a = Dense(13, activation='softmax', name="action")(x_1)
# predictions_b = Dense(40, activation='softmax', name="target")(x_1)    # 32
# predictions_c = Dense(7, activation='softmax', name="key")(x_1)
# predictions_d = Dense(144, activation='softmax', name="value")(x_1)    # 131
#
#
# model = Model(inputs=[inputs_a, inputs_b],
#               outputs=[predictions_a, predictions_b, predictions_c, predictions_d])


# model = getmodel()
(x_f_train, x_f_test, p_f_train, p_f_test, tfidf_f_train, tfidf_f_test, act_f_train, act_f_test, tar_f_train, tar_f_test, key_f_train, key_f_test, val_f_train, val_f_test) \
    = data01234[0]

print("训练...")

batch_size = 32
tensorboard = TensorBoard(log_dir="./log2/2")
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'], loss_weights=[0.2, 1.0, 0.5, 0.1])

model.fit([x_pry, p_pry, x_pry], [act_pry, tar_pry, key_pry, val_pry], batch_size=batch_size, epochs=10,
           verbose=1, callbacks=[tensorboard])

model.fit([x_f_train, p_f_train, x_f_train], [act_f_train, tar_f_train, key_f_train, val_f_train], batch_size=batch_size, epochs=16,
           verbose=1, callbacks=[tensorboard], \
           validation_data=([x_f_test, p_f_test, x_f_test], [act_f_test, tar_f_test, key_f_test, val_f_test]))

# --------------- validation for xgb --------------
# prob_train_pry = model.predict(x=[x_pry, p_pry, x_pry])
# act_prob_pry = prob_train_pry[0]
# tar_prob_pry = prob_train_pry[1]
# key_prob_pry = prob_train_pry[2]
# val_prob_pry = prob_train_pry[3]
# prob_train_f = model.predict(x=[x_f_train, p_f_train, x_f_train])
# act_prob_f = prob_train_f[0]
# tar_prob_f = prob_train_f[1]
# key_prob_f = prob_train_f[2]
# val_prob_f = prob_train_f[3]
#
# with open("train1.8_prob.pkl", "wb") as f:
#     pickle.dump((act_prob_pry, tar_prob_pry, key_prob_pry, val_prob_pry, act_prob_f, tar_prob_f, key_prob_f, val_prob_f), f)

# prob_test = model.predict(x=[x_f_test, p_f_test, x_f_test])
# act_pred = prob_test[0]
# tar_pred = prob_test[1]
# key_pred = prob_test[2]
# val_pred = prob_test[3]
#
# with open("test0.2_prob.pkl", "wb") as f:
#     pickle.dump((act_pred, tar_pred, key_pred, val_pred), f)
# model.save('model_weights0.h5')

# ---------------------------------------------
# predict_testB = model.predict(x=[text_test, positive_test, text_test])
# y1prob = predict_testB[0]
# y2prob = predict_testB[1]
# z1prob = predict_testB[2]
# z2prob = predict_testB[3]
#
# with open("testB0928_prob.pkl", "wb") as f:
#     pickle.dump((y1prob, y2prob, z1prob, z2prob), f)
# --------------- produce result ----------
# model.fit([x_final, p_final], [act_final, tar_final, key_final, val_final], batch_size=batch_size, epochs=3,
#            verbose=1, callbacks=[tensorboard])
#
# predict_label = model.predict(x=[text_test, positive_test])
#
# predict_action = predict_label[0]
# predict_target = predict_label[1]
# # predict_key = predict_label[2]
# # predict_value = predict_label[3]
#
# import pickle
# with open("yy0916.pkl", "wb") as f:
#     pickle.dump((predict_action, predict_target), f)

# ------------- for validation --------------
# model.fit([X_train, P_train], [action_train, target_train, key_train, value_train], batch_size=batch_size, epochs=16,
#            verbose=1, callbacks=[tensorboard], \
#           validation_data=([X_test, P_test], [action_test, target_test, key_test, value_test]))

# best epoch:  act-0.8320  tar-0.8215(epoch=3)    key-0.9578  value-0.9462(epoch=16)
