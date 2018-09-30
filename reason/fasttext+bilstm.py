# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Dense, Concatenate
from keras.models import Model
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.layers.pooling import AveragePooling2D
from keras.layers.core import Reshape
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split


def get_session(gpu_fraction=0.4):
    """
    This function is to allocate GPU memory a specific fraction
    Assume that you have 6GB of GPU memory and want to allocate ~2GB
    """

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# get_session(gpu_fraction=0.4)

np.random.seed(42)
label_num = [4, 37]
vocab_dim = 300
pos_dim = 30
vocabulary_size = 275456    # 词典数目：27545 + 填充符：1
pos_size = 275456   # 54


def build_input(input_dim, output_dim, shape):
    inputs = Input(shape=[shape[0], shape[1]])
    x = Reshape((shape[0] * shape[1],))(inputs)
    x = Embedding(output_dim=output_dim, input_dim=input_dim + 1, mask_zero=False,
                  input_length=shape[0] * shape[1])(x)
    x = Reshape((shape[0], shape[1], output_dim))(x)

    return inputs, x


def concat_output(x_right, x_left, vocab_dimension, pos_dimension, shape):
    x = Concatenate()([x_right, x_left])
    x = Dropout(0.5)(x)
    x = AveragePooling2D(data_format='channels_last', pool_size=(1, shape[1]))(x)
    x = Reshape((shape[0], vocab_dimension + pos_dimension))(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(GRU(300))(x)

    return x


train_dialogues, train_dialogues_pos, test_dialogues, test_dialogues_pos, y1_categorical, y2_categorical = pd.read_pickle(
    "xxxxyy.pkl")


def total_pad_seq(train_dialogues, length=60):
    tra = []
    for a in train_dialogues:
        a_1 = pad_sequences(a, maxlen=length, value=vocabulary_size + 1)
        tra.append(a_1)
    return np.array(tra)


x_train = total_pad_seq(train_dialogues)
x_pos_train = total_pad_seq(train_dialogues_pos)
X_test = total_pad_seq(test_dialogues)
X_pos_test = total_pad_seq(test_dialogues_pos)
y_train = y1_categorical
y1_train = y2_categorical

X_train_60 = pad_sequences(x_train, maxlen=60, value=vocabulary_size - 1)
# X_val_train_60 = pad_sequences(X_val_train, maxlen=60, value=vocabulary_size)
X_test_60 = pad_sequences(X_test, maxlen=60, value=vocabulary_size - 1)

X_pos_train_60 = pad_sequences(x_pos_train, maxlen=60, value=vocabulary_size - 1)
# X_val_pos_train_60 = pad_sequences(X_val_pos_train, maxlen=60, value = vocabulary_size+1)
X_pos_test_60 = pad_sequences(X_pos_test, maxlen=60, value=vocabulary_size - 1)

X_train_40 = pad_sequences(x_train, maxlen=40, value=vocabulary_size - 1)
# X_val_train_40 = pad_sequences(X_val_train, maxlen=40, value=vocabulary_size)
X_test_40 = pad_sequences(X_test, maxlen=40, value=vocabulary_size - 1)

X_pos_train_40 = pad_sequences(x_pos_train, maxlen=40, value=vocabulary_size - 1)
# X_val_pos_train_40 = pad_sequences(X_val_pos_train, maxlen=40, value = vocabulary_size+1)
X_pos_test_40 = pad_sequences(X_pos_test, maxlen=40, value=vocabulary_size - 1)

# ------
X_train_60, X_test_60, X_pos_train_60, X_pos_test_60, X_train_40, X_test_40, X_pos_train_40, X_pos_test_40, \
y_train, y_test, y1_train, y1_test = train_test_split(X_train_60, X_pos_train_60, X_train_40, X_pos_train_40, y_train, y1_train, test_size=0.2, random_state=42)

# ---------------------------

inputs_a, x_a = build_input(vocabulary_size, vocab_dim, X_train_60[0].shape)
inputs_b, x_b = build_input(pos_size, pos_dim, X_pos_train_60[0].shape)

inputs_c, x_c = build_input(vocabulary_size, vocab_dim, X_train_40[0].shape)
inputs_d, x_d = build_input(pos_size, pos_dim, X_pos_train_40[0].shape)

x_1 = concat_output(x_a, x_b, vocab_dim, pos_dim, X_train_60[0].shape)
x_2 = concat_output(x_c, x_d, vocab_dim, pos_dim, X_train_40[0].shape)

predictions_a = Dense(4, activation='softmax')(x_1)
predictions_b = Dense(37, activation='softmax')(x_1)

predictions_c = Dense(4, activation='softmax')(x_2)
predictions_d = Dense(37, activation='softmax')(x_2)

x = Concatenate()([x_1, x_2])
x = Dropout(0.5)(x)

predictions_e = Dense(4, activation='softmax')(x)
predictions_f = Dense(37, activation='softmax')(x)

model = Model(inputs=[inputs_a, inputs_b, inputs_c, inputs_d],
              outputs=[predictions_a, predictions_b, predictions_c, predictions_d, predictions_e, predictions_f])

# -------23轮 ：0.6012-------
print("训练...")

batch_size = 32
tensorboard = TensorBoard(log_dir="./log/final/101")
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'], loss_weights=[0.2, 1.0, 0.2, 1.0, 0.2, 1.0])


# ------------------------
# model.fit([X_train_60, X_pos_train_60, X_train_20, X_pos_train_20],
#           [y_train, y1_train, y_train, y1_train, y_train, y1_train],
#           batch_size=batch_size, epochs=23, verbose=1, callbacks=[tensorboard])

# ---------------
model.fit([X_train_60, X_pos_train_60, X_train_40, X_pos_train_40],
          [y_train, y1_train, y_train, y1_train, y_train, y1_train],
          batch_size=batch_size, epochs=32, verbose=1, callbacks=[tensorboard], \
          validation_data=([X_test_60, X_pos_test_60, X_test_40, X_pos_test_40], [y_test, y1_test, y_test, y1_test, y_test, y1_test]))

# -------
# predict_label = model.predict(x=[X_test_60, X_pos_test_60, X_test_40, X_pos_test_40])
#
# predict_y = predict_label[4]
# predict_y1 = predict_label[5]
#
# import pickle
#
# with open('predict_y.pkl', 'wb') as f:
#     pickle.dump(predict_y, f)
#
# with open('predict_y1.pkl', 'wb') as f:
#     pickle.dump(predict_y1, f)
# ----------------------

# import pickle
# with open('ID.pkl', 'rb') as f:
#     ID = pickle.load(f)
# y_dict, y1_dict = pd.read_pickle("yy_dict.pkl")
#
# print('id_num: '+str(len(ID)))
# print('y_num: '+str(len(predict_y)))
# print('y1_num: '+str(len(predict_y1)))
#
# resultStr = []
# for i in range(len(ID)):
#     str = ID[i] + "\t" + predict_y[i] + "\t" + predict_y1[i] + "\n"
#     resultStr.append(str)
#
# with open("./result.txt", mode="w", encoding="utf-8") as f:
#     f.writelines(resultStr)


#     pickle.dump(predict_y, f)
#
# with open('predict_y1.pkl', 'wb') as f:
#     pickle.dump(predict_y1, f)
