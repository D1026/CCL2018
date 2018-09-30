'''
This is an std multi-label classification
using myself metrics function
y format as:
y = [0, 0, 0, 1, 0 , 1, 0]

'''
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from sklearn.model_selection import train_test_split

np.random.seed(42)
label_num = [13, 40, 7, 74]
vocab_dim = 300
pos_dim = 30
vocabulary_size = 6949    # 6571
pos_size = 53    # 50


def build_input(input_dim, output_dim, shape):
    inputs = Input(shape=shape)
    x = Embedding(output_dim=output_dim, input_dim=input_dim + 1, mask_zero=False,
                    input_length=shape[0])(inputs)

    return inputs, x


def concat_output(x_right, x_left, vocab_dimension, pos_dimension):
    x = Concatenate()([x_right, x_left])
    x = Dropout(0.5)(x)
    x = AveragePooling1D(pool_size=1)(x)
    x = Reshape((-1, vocab_dimension + pos_dimension))(x)
    print(x.shape)
    x = Dropout(0.5)(x)
    x = Bidirectional(GRU(300))(x)

    return x


(text_train, positive_train, text_test, positive_test, action, target, key, value) = pd.read_pickle('xxxxyyzz.pkl')
# (intNum, sloNum) = pd.read_pickle('intNum_sloNum_oneHot.pkl')
(int, slo) = pd.read_pickle('multi_yz.pkl')
text_train, text_test, positive_train, positive_test, int_train, int_test, slo_train, slo_test = \
train_test_split(text_train, positive_train, int, slo, random_state=42, test_size=0.2)
# X_train, X_test, P_train, P_test, action_train, action_test, target_train, target_test, key_train, key_test, value_train, value_test\
#     = train_test_split(text, positive, action, target, key, value, random_state=42, shuffle=True, test_size=0.3)


inputs_a, x_a = build_input(vocabulary_size, vocab_dim, text_train[0].shape)
inputs_b, x_b = build_input(pos_size, pos_dim, positive_train[0].shape)


x_1 = concat_output(x_a, x_b, vocab_dim, pos_dim)

# ---------------------------
# predictions_a = Dense(13, activation='softmax', name="action")(x_1)
# predictions_b = Dense(40, activation='softmax', name="target")(x_1)
# predictions_c = Dense(7, activation='softmax', name="key")(x_1)
# predictions_d = Dense(74, activation='softmax', name="value")(x_1)
# ----------------------------
# predict_intNum = Dense(3, activation='softmax', name="intNum")(x_1)
# predict_sloNum = Dense(4, activation='softmax', name="sloNum")(x_1)
# ----------------------------
pre_int = Dense(47, activation='softmax', name="pre_int")(x_1)    # relu
pre_slo = Dense(194, activation='softmax', name="pre_slo")(x_1)
# -----------------------
# model = Model(inputs=[inputs_a, inputs_b],
#               outputs=[predictions_a, predictions_b, predictions_c, predictions_d])
# -----------------------
model = Model(inputs=[inputs_a, inputs_b],
              outputs=[pre_int, pre_slo])

print("训练...")
# ---------
import keras.backend as K

# def thrsh(x, val):
#     for i in x:
#         for j in range(len(i)):
#             if i[j] > val:
#                 i[j] = 1
#             else:
#                 i[j] = 0


def thresh_acc(y_true, y_pred):
    # test

    threshold = 0.466    # 0.95
    a = K.cast(K.greater(y_pred, threshold), K.floatx())
    b = K.cast(K.equal(a, y_true), K.floatx())
    c = K.equal(tf.reduce_sum(b, axis=-1), tf.reduce_sum(K.ones_like(b), axis=-1))
    return K.mean(K.cast(c, K.floatx()), axis=-1)

batch_size = 32
tensorboard = TensorBoard(log_dir="./log2/2")
model.compile(loss='categorical_crossentropy',   # binary_crossentropy
              optimizer='RMSprop',
              metrics=[thresh_acc])
              # loss_weights=[0.2, 1.0, 0.5, 0.1])
# validation_data=([X_test, P_test], [action_test, target_test, key_test, value_test]),
# ------------------------------
# model.fit([text_train, positive_train], [action, target, key, value], batch_size=batch_size, epochs=32,
#            verbose=1, callbacks=[tensorboard])
# ------------------------------
model.fit([text_train, positive_train], [int_train, slo_train], batch_size=batch_size, epochs=32,
           verbose=1, callbacks=[tensorboard], \
          validation_data=([text_test, positive_test], [int_test, slo_test]))

# 25: val_intNum_acc: 0.9703 - val_sloNum_acc: 0.9708
# ---------------------------------------
# predict_label = model.predict(x=[text_test, positive_test])
#
# predict_action = predict_label[0]
# predict_target = predict_label[1]
# predict_key = predict_label[2]
# predict_value = predict_label[3]
#
# import pickle
# with open("yyzz.pkl", "wb") as f:
#     pickle.dump((predict_action, predict_target, predict_key, predict_value), f)