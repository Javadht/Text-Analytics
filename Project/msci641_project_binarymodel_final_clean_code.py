# -*- coding: utf-8 -*-
"""MSCI641-Project-binarymodel-Final Clean Code.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dlzGyss_JcviQUfqcvzVMcreISmSvuyD
"""

import re
import numpy as np
import gensim
import pandas as pd
from keras.layers import GRU
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from numpy import array
from tensorflow import keras
from tensorflow.keras import regularizers
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, BatchNormalization, Activation, Bidirectional
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.models import load_model
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive')

"""Configurations"""

# Max number of words in a sequence
max_len_head = 20
max_len_body = 80

"""Import Data"""

headers_train = pd.read_csv('/content/drive/My Drive/FNC-Project/train_stances.csv')
bodies_train = pd.read_csv('/content/drive/My Drive/FNC-Project/train_bodies.csv')
headers_test = pd.read_csv('/content/drive/My Drive/FNC-Project/competition_test_stances.csv')
bodies_test = pd.read_csv('/content/drive/My Drive/FNC-Project/test_bodies.csv')

"""***Filter Training Data*** Just for partially trained models"""

headers_train = headers_train[headers_train['Stance'] != 'unrelated']
headers_train = headers_train.reset_index(drop = True)

"""**Cleaning and Tokenizing data and remove punctuation and Stop words**"""

import string
from nltk.corpus import stopwords
import nltk
from collections import Counter
from keras.preprocessing.text import Tokenizer
nltk.download('stopwords')

stop = set(stopwords.words("english"))

def remove_punct(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)


def remove_stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in stop]

    return " ".join(text)

def counter_word(text):
    count = Counter()
    for i in text.values:
        for word in i.split():
            count[word] += 1
    return count

headers_test["Headline"] = headers_test.Headline.map(lambda x: remove_punct(x))
headers_train["Headline"] = headers_train.Headline.map(lambda x: remove_punct(x))
bodies_train["articleBody"] = bodies_train.articleBody.map(lambda x: remove_punct(x))
bodies_test["articleBody"] = bodies_test.articleBody.map(lambda x: remove_punct(x))


headers_test["Headline"]= headers_test["Headline"].map(remove_stopwords)
headers_train["Headline"]= headers_train["Headline"].map(remove_stopwords)
bodies_train["articleBody"]= bodies_train["articleBody"].map(remove_stopwords)
bodies_test["articleBody"]= bodies_test["articleBody"].map(remove_stopwords)

#Tokenizing training sets
text = headers_train.Headline.append(bodies_train.articleBody) 
counter = counter_word(text)
text = text.values.tolist()
text_token = [line.split() for line in text]
num_words = len(counter)

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(headers_train.Headline.append(bodies_train.articleBody))
word_index = tokenizer.word_index

train_sequences_header = tokenizer.texts_to_sequences(headers_train.Headline)
train_sequences_body = tokenizer.texts_to_sequences(bodies_train.articleBody)

"""**Building Word2Vec model** with training set"""

#Based on choice, one of this cell or the next one should be run
w2vmodel = Word2Vec(
        text_token,
        size=50,
        window=5,
        min_count=1,
        workers=4,
    )
W2vec_Status = 'with training set'

"""**Building Word2Vec model** by importing Glove"""

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath('/content/drive/My Drive/FNC-Project/glove.6B.50d.txt')
tmp_file = get_tmpfile("test_word2vec.txt")

_ = glove2word2vec(glove_file, tmp_file)

w2vmodel = KeyedVectors.load_word2vec_format(tmp_file)

W2vec_Status = 'Imported from the Glove.6B.50d'

"""**Preparing Embedding matrix**"""

embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(tokenizer.word_index) + 1, 50))

for word, i in tokenizer.word_index.items():
    try:
        embeddings_vector = w2vmodel[word]
    except KeyError:
        embeddings_vector = None
    if embeddings_vector is not None:
        embeddings_matrix[i] = embeddings_vector

"""**Creating input Data** Sequencing and padding"""

from keras.preprocessing.sequence import pad_sequences

train_padded_head = pad_sequences(
    train_sequences_header, maxlen=max_len_head, padding="post", truncating="post")
train_padded_body = pad_sequences(
    train_sequences_body, maxlen=max_len_body, padding="post", truncating="post")

train_padded = np.zeros((len(headers_train),max_len_head+max_len_body),dtype = 'i')
for i in tqdm(range(0, len(headers_train),1)):
  BodyID = headers_train["Body ID"][i]
  j = bodies_train[bodies_train["Body ID"] == BodyID].index
  train_padded[i] = np.append(train_padded_head[i] ,train_padded_body[j])
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

"""**Preparing Labels**"""

#prepare labels based on the desired output of the model 
labels1 = pd.DataFrame()
headers_train.loc[headers_train['Stance'] == 'unrelated', 'stance_id'] = 1
headers_train.loc[headers_train['Stance'] == 'agree', 'stance_id'] = 2
headers_train.loc[headers_train['Stance'] == 'disagree', 'stance_id'] = 3
headers_train.loc[headers_train['Stance'] == 'discuss', 'stance_id'] = 4

label_test = headers_train
one_hot = pd.get_dummies(label_test['stance_id'])
label_test = label_test.join(one_hot)
x = label_test.filter(items = [1.0,2.0, 3.0, 4.0], axis = 1)
labels = x.to_numpy()
labels = labels.reshape(len(headers_train),4)

"""Building model"""

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten
from keras.initializers import Constant
from keras.optimizers import Adam , SGD


model1 = Sequential()
model1.add(Embedding(len(tokenizer.word_index)+1 ,50, weights = [embeddings_matrix], input_length=max_len_head + max_len_body))

model1.add(LSTM(256, dropout=0.5))
model1.add(Dense(128,activation = "relu", activity_regularizer=regularizers.l2(0.01)))
model1.add(Dropout (0.2))
model1.add(Dense(32,activation = "relu", activity_regularizer=regularizers.l2(0.01)))
model1.add(Dropout (0.2))
model1.add(Dense(4, activation="softmax"))


#optimizer = SGD(lr = 1e-3, momentum = 0.9, decay = 0.01)
optimizer = Adam(learning_rate=3e-4)
model1.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model1.summary()

model = model1.fit(
    train_padded, labels,batch_size = 64 ,epochs=10, validation_data=(train_padded, labels),
)

model1.save('/content/drive/My Drive/FNC-Project/model400-1.model')