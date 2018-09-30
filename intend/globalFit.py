import pandas as pd
import numpy as np
from numpy import random
import pickle
from keras import optimizers
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
vocabulary_size = 16288
pos_size = 54

embedding_matrix = pd.read_pickle('embedding_matrix.pkl')


def build_input(input_dim, output_dim, shape):
    inputs = Input(shape=shape)
    x = Embedding(output_dim=output_dim, input_dim=input_dim + 1, mask_zero=False,
                    input_length=shape[0])(inputs)

    return inputs, x


def vec_input(shape):
    inputs = Input(shape=shape)
    x = Embedding(input_dim=16288, output_dim=100, weights=[embedding_matrix], mask_zero=False,
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
# (tfidf_train, tfidf_test) = pd.read_pickle('tfidf_train_test.pkl')
print(len(action))

# ----- 预测对象 -----！！！
x_fB = text_test[:10000]
p_fB = positive_test[:10000]


# --------切割初赛复赛数据集--------
(x_pry, p_pry, act_pry, tar_pry, key_pry, val_pry) = pd.read_pickle("shuffled_pry.pkl")
(X_train, P_train, Y1_train, Y2_train, Z1_train, Z2_train) = pd.read_pickle('xpy1y2z1z2_train.pkl')
(X_vali, P_vali, Y1_vali, Y2_vali, Z1_vali, Z2_vali) = pd.read_pickle("xpy1y2z1z2_vali.pkl")

X_final = X_train + X_vali
P_final = P_train + P_vali
Y1_final = Y1_train + Y1_vali
Y2_final = Y2_train + Y2_vali
Z1_final = Z1_train + Z1_vali
Z2_final = Z2_train + Z2_vali
print('X_train len: ', len(X_train), type(X_train))
print('X_vali len:  ', len(X_vali), type(X_vali))
# fake label data
(text_test, positive_test, fake_act, fake_tar, fake_key, fake_val) = pd.read_pickle('fake_train_xpatkv.pkl')

# --------------------for validation------------------


def getmodel(vocabulary_size = 16288, pos_size = 54, vocab_dim = 512, pos_dim = 32):
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
print("训练...")
sgd = optimizers.SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=True)

batch_size = 32
tensorboard = TensorBoard(log_dir="./log2/2")

model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'], loss_weights=[0.5, 0.75, 0.5, 0.25])

for i in range(3):
    np.random.seed(i+8)

    model.fit([text_test, positive_test, text_test], [fake_act, fake_tar, fake_key, fake_val], batch_size=batch_size, epochs=1,
              verbose=1, shuffle=True, validation_split=0.1)

    model.fit([x_pry, p_pry, x_pry], [act_pry, tar_pry, key_pry, val_pry], batch_size=batch_size, epochs=1,
              verbose=1, shuffle=True, validation_split=0.05)

    print('当前 i = ', i)

    # model.fit([X_train, P_train, X_train], [Y1_train, Y2_train, Z1_train, Z2_train], batch_size=batch_size,
    #           epochs=3,
    #           verbose=1,
    #           validation_data=([X_vali, P_vali, X_vali], [Y1_vali, Y2_vali, Z1_vali, Z2_vali]))

    # --------------------------------------
    model.fit([X_final, P_final, X_final], [Y1_final, Y2_final, Z1_final, Z2_final], batch_size=batch_size,
              epochs=3,
              verbose=1,
              )
    # model.save('model_Ivan'+str(i))

np.random.seed(11)

model.fit([text_test, positive_test, text_test], [fake_act, fake_tar, fake_key, fake_val], batch_size=batch_size, epochs=1,
          verbose=1, shuffle=True, validation_split=0.1)

model.fit([x_pry, p_pry, x_pry], [act_pry, tar_pry, key_pry, val_pry], batch_size=batch_size, epochs=1,
          verbose=1, shuffle=True, validation_split=0.05)


model.fit([X_final, P_final, X_final], [Y1_final, Y2_final, Z1_final, Z2_final], batch_size=batch_size,
          epochs=2,
          verbose=1,
          shuffle=True)

# ----------------------------
def softmaxToOnehont(y):
    shape = y.shape
    y_ = np.zeros(shape=shape)
    for i in range(shape[0]):
        index = np.argmax(y[i])
        y_[i][index] = 1
    return y_

fakeLebel = model.predict(x=[text_test, positive_test, text_test])  # 12 5002 个

fake_act = softmaxToOnehont(fakeLebel[0])
fake_tar = softmaxToOnehont(fakeLebel[1])
fake_key = softmaxToOnehont(fakeLebel[2])
fake_val = softmaxToOnehont(fakeLebel[3])

print('fake_act length: ', len(fake_act))

with open("fake_train_xpatkv.pkl", "wb") as f:
    pickle.dump((text_test, positive_test, fake_act, fake_tar, fake_key, fake_val), f)
