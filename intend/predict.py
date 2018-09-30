import pandas as pd
import numpy as np
from numpy import random
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import *

# ----------------------------------------------------------
# (text_train, positive_train, text_test, positive_test, action, target, key, value) = pd.read_pickle('xxxxyyzz.pkl')
# model0 = load_model('model_weights0.h5')
# model1 = load_model('model_weights1.h5')
# model2 = load_model('model_weights2.h5')
# model3 = load_model('model_weights3.h5')
# model4 = load_model('model_weights4.h5')

# ------------------------------------------------------------
# pre0 = model0.predict(x=[text_test, positive_test, text_test])
# pre1 = model1.predict(x=[text_test, positive_test, text_test])
# pre2 = model2.predict(x=[text_test, positive_test, text_test])
# pre3 = model3.predict(x=[text_test, positive_test, text_test])
# pre4 = model4.predict(x=[text_test, positive_test, text_test])

# pre0 = model0.predict(x=[text_train, positive_train, text_train])
# pre1 = model1.predict(x=[text_train, positive_train, text_train])
# pre2 = model2.predict(x=[text_train, positive_train, text_train])
# pre3 = model3.predict(x=[text_train, positive_train, text_train])
# pre4 = model4.predict(x=[text_train, positive_train, text_train])

(act_A, tar_A, key_A, val_A) = pd.read_pickle('atkv0.pkl')
(act_B, tar_B, key_B, val_B) = pd.read_pickle('atkv1.pkl')
(act_C, tar_C, key_C, val_C) = pd.read_pickle('atkv2.pkl')
(act_D, tar_D, key_D, val_D) = pd.read_pickle('atkv3.pkl')

print(type(act_A))


# act_B = np.array(pre1[0])
# tar_B = np.array(pre1[1])
# key_B = np.array(pre1[2])
# val_B = np.array(pre1[3])
#
# act_C = np.array(pre2[0])
# tar_C = np.array(pre2[1])
# key_C = np.array(pre2[2])
# val_C = np.array(pre2[3])
#
# act_D = np.array(pre3[0])
# tar_D = np.array(pre3[1])
# key_D = np.array(pre3[2])
# val_D = np.array(pre3[3])
#
# act_E = np.array(pre4[0])
# tar_E = np.array(pre4[1])
# key_E = np.array(pre4[2])
# val_E = np.array(pre4[3])

act_pre = act_A + act_B + act_C + act_D
tar_pre = tar_A + tar_B + tar_C + tar_D
key_pre = key_A + key_B + key_C + key_D
val_pre = val_A + val_B + val_C + val_D

with open("atkv0929.pkl", "wb") as f:
    pickle.dump((act_pre, tar_pre, key_pre, val_pre), f)
