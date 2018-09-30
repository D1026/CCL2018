import json
import pickle
from jieba import posseg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


with open('intent_data.train.0903', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
train_samples = []
for i in lines:
    train_samples.append(json.loads(i.split('\t')[1]))

with open('intent_data.recharge.testA.0903', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
test_samples = []
for i in lines:
    test_samples.append(json.loads(i.split('\t')[1]))

train_sentences = []
test_sentences = []
sentences = []
for sp in train_samples:
    train_sentences.append(sp['sentence'])
for sp in test_samples:
    test_sentences.append(sp['sentence'])

train_chars = []
for i in train_sentences:
    train_chars.append(list(i))

test_chars = []
for i in test_sentences:
    test_chars.append(list(i))

chars = train_chars + test_chars

tokenizer = Tokenizer(num_words=30000)
tokenizer.fit_on_texts(chars)
char_symbols = len(tokenizer.word_index) + 1

train_x = tokenizer.texts_to_sequences(train_chars)
test_x = tokenizer.texts_to_sequences(test_chars)

print(tokenizer.word_index)
print(chars[0])
print(train_x[0])
print('train_chars length:', len(train_chars))
print('test_x length: ', len(test_x))
print(char_symbols)    # 1983

train_x = pad_sequences(train_x, maxlen=32)
test_x = pad_sequences(test_x, maxlen=32)

import pickle
with open('xx_chars.pkl', 'wb') as f:
    pickle.dump((train_x, test_x), f)
