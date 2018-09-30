# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Conv1D, MaxPooling2D, AveragePooling1D
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import Reshape


np.random.seed(42)
char_len = 32
char_dim = 128
char_size = 1983


def build_input(input_dim, output_dim, len):
    inputs = Input(shape=(len,))
    x = Embedding(output_dim=output_dim, input_dim=input_dim, mask_zero=False,
                  input_length=len)(inputs)
    x = Reshape((len, output_dim))(x)
    x = Dropout(0.5)(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same", input_shape=(len, output_dim), activation="relu")(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(GRU(300))(x)
    return inputs, x


train_dialogues, test_dialogues = pd.read_pickle("xx_character_level.pkl")


def total_pad_seq(train_dialogues, length=60):
    tra = []
    for a in train_dialogues:
        a_1 = pad_sequences(a, maxlen=length, value=vocabulary_size + 1)
        tra.append(a_1)
    return np.array(tra)


x_train = total_pad_seq(train_dialogues)
X_test = total_pad_seq(test_dialogues)


X_train_60 = pad_sequences(x_train, maxlen=60, value=vocabulary_size + 1)
X_test_60 = pad_sequences(X_test, maxlen=60, value=vocabulary_size + 1)


X_train_40 = pad_sequences(x_train, maxlen=40, value=vocabulary_size)
X_test_40 = pad_sequences(X_test, maxlen=40, value=vocabulary_size)


inputs_a, x_a = build_input(vocabulary_size, vocab_dim, X_train_60[0].shape)
print(x_a.shape)
inputs_b, x_b = build_input(vocabulary_size, vocab_dim, X_train_40[0].shape)
print(x_b.shape)



