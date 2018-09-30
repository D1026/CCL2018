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