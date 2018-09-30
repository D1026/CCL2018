import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from sklearn.model_selection import train_test_split

np.random.seed(42)
label_num = [13, 32, 7, 131]    # [13, 40, 7, 74]
vocab_dim = 300
pos_dim = 32
char_dim = 128
vocabulary_size = 6949    # 6571
pos_size = 53    # 50
char_size = 1983
char_len = 32



def build_input(input_dim, output_dim, shape):
    inputs = Input(shape=shape)
    x = Embedding(output_dim=output_dim, input_dim=input_dim + 1, mask_zero=False,
                    input_length=shape[0])(inputs)

    return inputs, x


def chars_input(input_dim, output_dim, len):
    inputs = Input(shape=(len,))
    x = Embedding(output_dim=output_dim, input_dim=input_dim, mask_zero=False,
                  input_length=len)(inputs)
    x = Reshape((len, output_dim))(x)
    x = Dropout(0.5)(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same", input_shape=(len, output_dim), activation="relu")(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    # ------------------
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    # ------------------
    # x = Bidirectional(LSTM(31))(x)
    # x = Bidirectional(LSTM(300))(x)
    return inputs, x


# -----------------
# def concat_output(x_word, x_pos, x_char, vocab_dimension, pos_dimension):
#     print(x_char)
#     x = Concatenate()([x_word, x_pos])
#     x = Dropout(0.5)(x)
#     x = AveragePooling1D(pool_size=1)(x)
#     x = Reshape((-1, vocab_dimension + pos_dimension))(x)
#     x_char = Reshape((31, 2))(x_char)
#     x = Concatenate()([x, x_char])
#     print(x.shape)
#     x = Dropout(0.5)(x)
#     x = Bidirectional(LSTM(31))(x)
#
#     return x


def concat_output(x_word, x_pos, vocab_dimension, pos_dimension):
    x = Concatenate()([x_word, x_pos])
    x = Dropout(0.5)(x)
    x = AveragePooling1D(pool_size=1)(x)
    x = Reshape((-1, vocab_dimension + pos_dimension))(x)
    print(x.shape)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(300))(x)

    return x


(text_train, positive_train, text_test, positive_test, action, target, key, value) = pd.read_pickle('xxxxyyzz.pkl')
(char_train, char_test) = pd.read_pickle('xx_chars.pkl')
# ---------- for test ----------
X_train, X_test, P_train, P_test, C_train, C_test, action_train, action_test, target_train, target_test, key_train, key_test, value_train, value_test\
    = train_test_split(text_train, positive_train, char_train, action, target, key, value, random_state=42, test_size=0.3)


inputs_a, x_a = build_input(vocabulary_size, vocab_dim, text_train[0].shape)
inputs_b, x_b = build_input(pos_size, pos_dim, positive_train[0].shape)
inputs_c, x_c = chars_input(char_size, char_dim, char_len)


# x_1 = concat_output(x_a, x_b, x_c, vocab_dim, pos_dim)
x_1 = concat_output(x_a, x_b, vocab_dim, pos_dim)
x_2 = Concatenate()([x_1, x_c])

predictions_a = Dense(13, activation='softmax', name="action")(x_2)
predictions_b = Dense(32, activation='softmax', name="target")(x_2)
predictions_c = Dense(7, activation='softmax', name="key")(x_2)
predictions_d = Dense(131, activation='softmax', name="value")(x_2)


model = Model(inputs=[inputs_a, inputs_b, inputs_c],
              outputs=[predictions_a, predictions_b, predictions_c, predictions_d])

print("训练...")

batch_size = 32
tensorboard = TensorBoard(log_dir="./log2/2")
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'], loss_weights=[0.2, 1.0, 0.5, 0.1])

model.fit([X_train, P_train, C_train], [action_train, target_train, key_train, value_train], batch_size=batch_size, epochs=32,
           verbose=1, callbacks=[tensorboard], \
          validation_data=([X_test, P_test, C_test], [action_test, target_test, key_test, value_test]))

# best 15