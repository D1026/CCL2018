# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.layers.embeddings import Embedding
from keras.layers.core import *
from keras.layers import *
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.layers.pooling import AveragePooling2D
from keras.layers.core import Reshape
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import keras.backend as K
import time

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
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


# vocabulary_size = 27545
# pos_size = 54

"""
def build_char_input(input_dim, output_dim, shape):
    inputs = Input(shape=[shape[0], shape[1]])
    x = Conv1D(filters=shape[1] * 2, kernel_size=2, padding="same", input_shape=(shape[0], shape[1]), activation="relu")(inputs)
    x = AveragePooling1D(pool_size=2)(x)
    x = Reshape((shape[0], shape[1]))(x)
    x = Reshape((shape[0] * shape[1],))(x)
    x = Embedding(output_dim=output_dim, input_dim=input_dim + 1, mask_zero=False,
                  input_length=shape[0] * shape[1])(x)
    x = Reshape((shape[0], shape[1], output_dim))(x)
    return inputs, x
"""


class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def build_input(input_dim, output_dim, shape):
    inputs = Input(shape=[shape[0], shape[1]])
    x = Reshape((shape[0] * shape[1],))(inputs)
    x = Embedding(output_dim=output_dim, input_dim=input_dim + 1, mask_zero=False,
                  input_length=shape[0] * shape[1])(x)
    # x = BatchNormalization()(x)
    x = Reshape((shape[0], shape[1], output_dim))(x)

    return inputs, x


def concat_output(x_train, x_train_pos, vocab_dimension, pos_dimension, shape):
    x = Concatenate()([x_train, x_train_pos])
    # x = BatchNormalization()(x)
    x = AveragePooling2D(data_format='channels_last', pool_size=(1, shape[1]))(x)
    # x = BatchNormalization()(x)
    x = Reshape((shape[0], vocab_dimension + pos_dimension))(x)
    x = Dropout(0.3)(x)
    # x = BatchNormalization()(x)
    x = Bidirectional(GRU(300, return_sequences=True))(x)
    x = AttLayer(300)(x)
    return x


def total_pad_seq(train_dialogues, length=60):
    tra = []
    for a in train_dialogues:
        a_1 = pad_sequences(a, maxlen=length, value=vocabulary_size - 1)
        tra.append(a_1)
    return np.array(tra)

# train_dialogues, train_dialogues_pos, testA_dialogues, testA_dialogues_pos, train_dialogues_original, train_dialogues_pos_original, \
# unlabeled_dialogues, unlabele`d_dialogues_pos = pd.read_pickle("xxxxxxxx.pkl")
train_dialogues, train_dialogues_pos, train_dialogues_original, train_dialogues_pos_original, testA_dialogues, testA_dialogues_pos,\
testB_dialogues, testB_dialogues_pos, testC_dialogues, testC_dialogues_pos, testD_dialogues, testD_dialogues_pos,\
unlabeled_dialogues, unlabeled_dialogues_pos = pd.read_pickle("14x.pkl")
y_class, y1_class, train_y, train_y1, ori_train_y, ori_train_y1, test_y, test_y1, unlabeled_y, unlabeled_y1 = pd.read_pickle("yyyyyyyy_encoder.pkl")
vocab_dim = 300
pos_dim = int(vocab_dim/5)
vocabulary_size = 67893


X_train = total_pad_seq(train_dialogues)
X_pos_train = total_pad_seq(train_dialogues_pos)


X_testA = total_pad_seq(testA_dialogues)
X_pos_testA = total_pad_seq(testA_dialogues_pos)

X_testD = total_pad_seq(testD_dialogues)
X_pos_testD = total_pad_seq(testD_dialogues_pos)


X_train_original = total_pad_seq(train_dialogues_original)
X_pos_train_original = total_pad_seq(train_dialogues_pos_original)

X_unlabeled = total_pad_seq(unlabeled_dialogues)
X_pos_unlabeled = total_pad_seq(unlabeled_dialogues_pos)



X_train_60 = pad_sequences(X_train, maxlen=60, value=vocabulary_size)
X_train_ori_60 = pad_sequences(X_train_original, maxlen=60, value=vocabulary_size)
X_testA_60 = pad_sequences(X_testA, maxlen=60, value=vocabulary_size)
X_testD_60 = pad_sequences(X_testD, maxlen=60, value=vocabulary_size)
X_unlabeled_60 = pad_sequences(X_unlabeled, maxlen=60, value=vocabulary_size)

X_pos_train_60 = pad_sequences(X_pos_train, maxlen=60, value=vocabulary_size)
X_pos_train_ori_60 = pad_sequences(X_pos_train_original, maxlen=60, value=vocabulary_size)
X_pos_testA_60 = pad_sequences(X_pos_testA, maxlen=60, value=vocabulary_size)
X_pos_testD_60 = pad_sequences(X_pos_testD, maxlen=60, value=vocabulary_size)
X_pos_unlabeled_60 = pad_sequences(X_pos_unlabeled, maxlen=60, value=vocabulary_size)

X_train_40 = pad_sequences(X_train, maxlen=40, value=vocabulary_size)
X_train_ori_40 = pad_sequences(X_train_original, maxlen=40, value=vocabulary_size)
X_testA_40 = pad_sequences(X_testA, maxlen=40, value=vocabulary_size)
X_testD_40 = pad_sequences(X_testD, maxlen=40, value=vocabulary_size)
X_unlabeled_40 = pad_sequences(X_unlabeled, maxlen=40, value=vocabulary_size)

X_pos_train_40 = pad_sequences(X_pos_train, maxlen=40, value=vocabulary_size)
X_pos_train_ori_40 = pad_sequences(X_pos_train_original, maxlen=40, value=vocabulary_size)
X_pos_testA_40 = pad_sequences(X_pos_testA, maxlen=40, value=vocabulary_size)
X_pos_testD_40 = pad_sequences(X_pos_testD, maxlen=40, value=vocabulary_size)
X_pos_unlabeled_40 = pad_sequences(X_pos_unlabeled, maxlen=40, value=vocabulary_size)

inputs_a, x_a = build_input(vocabulary_size, vocab_dim, X_train_60[0].shape)
inputs_b, x_b = build_input(vocabulary_size, pos_dim, X_pos_train_60[0].shape)
# inputs_c, x_c = build_input(vocabulary_size, vocab_dim, X_train_ori_60[0].shape)
# inputs_d, x_d = build_input(vocabulary_size, vocab_dim, X_pos_train_ori_60[0].shape)

inputs_e, x_e = build_input(vocabulary_size, vocab_dim, X_train_40[0].shape)
inputs_f, x_f = build_input(vocabulary_size, pos_dim, X_pos_train_40[0].shape)
# inputs_g, x_g = build_input(vocabulary_size, vocab_dim, X_train_ori_40[0].shape)
# inputs_h, x_h = build_input(vocabulary_size, vocab_dim, X_pos_train_ori_40[0].shape)


x_1 = concat_output(x_a, x_b, vocab_dim, pos_dim, X_train_60[0].shape)
x_2 = concat_output(x_e, x_f, vocab_dim, pos_dim, X_train_40[0].shape)
# x_1 = BatchNormalization()(x_1)
# x_2 = BatchNormalization()(x_2)

predictions_a = Dense(5, activation='softmax')(x_1)
x_1_with_prediction_a = Concatenate()([x_1, predictions_a])
predictions_b = Dense(38, activation='softmax')(x_1_with_prediction_a)


predictions_c = Dense(5, activation='softmax')(x_2)
x_2_with_prediction_c = Concatenate()([x_1, predictions_a])
predictions_d = Dense(38, activation='softmax')(x_2_with_prediction_c)

x = Concatenate()([x_1, x_2])
x = Dropout(0.3)(x)
# x = BatchNormalization()(x)

predictions_e = Dense(5, activation='softmax')(x)
x_with_prediction_e = Concatenate()([x, predictions_e])
predictions_f = Dense(38, activation='softmax')(x_with_prediction_e)

model = Model(inputs=[inputs_a, inputs_b, inputs_e, inputs_f],
              outputs=[predictions_a, predictions_b, predictions_c, predictions_d, predictions_e, predictions_f])

print("训练...")
from keras import optimizers
sgd = optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
batch_size = 32
tensorboard = TensorBoard(log_dir="./log/10")
x_train_60, x_val_60, x_pos_train_60, x_val_pos_60, x_train_ori_60, x_val_ori_60, x_pos_ori_60, x_val_pos_ori_60, \
x_train_40, x_val_40, x_pos_train_40, x_val_pos_40, x_train_ori_40, x_val_ori_40, x_pos_ori_40, x_val_pos_ori_40,\
y, y_val, y1, y1_val = train_test_split(X_train_60, X_pos_train_60, X_train_ori_60, X_pos_train_ori_60, X_train_40, X_pos_train_40, X_train_ori_40, X_pos_train_ori_40, train_y, train_y1, test_size=0.1, random_state=30)

print('**×*×*×*×*×*×*×*×*×*×*×*×*×*×*')
model.load_weights('model_new_leo.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'], loss_weights=[0.2, 1.0, 0.2, 1.0, 0.2, 1.0])

for i in range(10):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    history1 = model.fit([X_unlabeled_60, X_pos_unlabeled_60, X_unlabeled_40, X_pos_unlabeled_40],
                         [unlabeled_y, unlabeled_y1, unlabeled_y, unlabeled_y1, unlabeled_y, unlabeled_y1],
                         batch_size=batch_size, epochs=1, verbose=1)
    history = model.fit([x_train_60, x_pos_train_60, x_train_40, x_pos_train_40],
              [y, y1, y, y1, y, y1],
              validation_data=([x_val_60, x_val_pos_60, x_val_40, x_val_pos_40], [y_val, y1_val, y_val, y1_val, y_val, y1_val]),
              batch_size=batch_size, epochs=2, verbose=1, callbacks=[tensorboard])

    for i in history.history.keys():
        if "acc" in i:
            print(i, history.history[i])
model.save("model_new_leo+10.h5")

exit()
predict_label_test = model.predict(x=[X_testD_60, X_pos_testD_60, X_testD_40, X_pos_testD_40])

predict_label_test_y = predict_label_test[0]
predict_label_test_y1 = predict_label_test[1]

import pickle
with open('predicted_test_9_26_23_00.pkl', 'wb') as f:
    pickle.dump((predict_label_test_y, predict_label_test_y1), f)


[0.509500003695488, 0.5260000022649765, 0.5215000004768372, 0.5369999985098839, 0.534000002026558, 0.5210000008940697, 0.5340000013113022, 0.5390000005364418, 0.5305000009536743, 0.5290000017881393, 0.5410000007748604, 0.5250000022649765, 0.5385000007748604, 0.5380000039339066, 0.5370000001192093, 0.5534999990463256, 0.53549999833107, 0.5440000005960465, 0.5385000015497208, 0.5415000011920929]
[0.49600000321865084, 0.5245000038146973, 0.5195000010728836, 0.5075000039935113, 0.5280000011920929, 0.5355000025033951, 0.5265000001192093, 0.543500003695488, 0.5220000010728836, 0.5265000026226043, 0.5145000023841858, 0.5290000005960465, 0.5279999994039536, 0.5335000010728836, 0.5270000015497207, 0.5375000033378601, 0.5295000001192093, 0.5335000016689301, 0.5270000040531159, 0.5315000011920928]
