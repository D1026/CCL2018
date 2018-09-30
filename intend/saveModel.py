import pandas as pd
import numpy as np
from numpy import random
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from intend.attenLayer import AttLayer
from intend.model import getmodel
from sklearn.model_selection import train_test_split


np.random.seed(42)
data01234 = pd.read_pickle('data01234.pkl')
(x_pry, p_pry, tfidf_pry, act_pry, tar_pry, key_pry, val_pry) = pd.read_pickle('shuffled_pry.pkl')

(text_train, positive_train, text_test, positive_test, action, target, key, value) = pd.read_pickle('xxxxyyzz.pkl')
text_test = text_test[:10000]
positive_test = positive_test[:10000]

model = getmodel()
print(model.get_weights())

# ------------ change ------------
(x_f_train, x_f_test, p_f_train, p_f_test, tfidf_f_train, tfidf_f_test, act_f_train, act_f_test, tar_f_train, tar_f_test, \
 key_f_train, key_f_test, val_f_train, val_f_test) = data01234[4]
# -----------

print("训练...")

batch_size = 32
tensorboard = TensorBoard(log_dir="./log2/2")
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'], loss_weights=[0.2, 1.0, 0.5, 0.1])

model.fit([x_pry, p_pry, x_pry], [act_pry, tar_pry, key_pry, val_pry], batch_size=batch_size, epochs=8,
           verbose=1, callbacks=[tensorboard])

model.fit([x_f_train, p_f_train, x_f_train], [act_f_train, tar_f_train, key_f_train, val_f_train], batch_size=batch_size, epochs=16,
           verbose=1, callbacks=[tensorboard], \
           validation_data=([x_f_test, p_f_test, x_f_test], [act_f_test, tar_f_test, key_f_test, val_f_test]))

# --------------- change ----------
# predict = model.predict(x=[text_test, positive_test, text_test])
# act = predict[0]
# tar = predict[1]
# keyy = predict[2]
# val = predict[3]
#
# with open("atkv4.pkl", "wb") as f:
#     pickle.dump((act, tar, keyy, val), f)
#
# model.save_weights('weights4.h5')

# bset epoch: [6, 5, 5, 9, ]