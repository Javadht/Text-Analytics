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

pos = open("pos.txt", 'r', errors='ignore')
postext = pos.readlines()
neg = open("neg.txt", 'r', errors='ignore')
negtext = neg.readlines()

## Specify Hyperparameters:
Max_Sent_len = 30
Max_Vocab_size = 20000
LSTM_dim = 128
EMBEDDING_dim = 100
Batch_Size = 48
N_Epochs = 2

df = pd.DataFrame(columns=['Comment', 'Label'])
df['Comment'] = postext + negtext
df['Label'] = [1] * len(postext) + [0] * len(negtext)
df = df.sample(frac=1, random_state=10)  # Shuffle the rows
df.reset_index(inplace=True, drop=True)
word_seq = [text_to_word_sequence(sent) for sent in df['Comment']]
tokenizer = Tokenizer(num_words=Max_Vocab_size)
tokenizer.fit_on_texts([' '.join(seq[:Max_Sent_len]) for seq in word_seq])

with open('train.csv') as f:
    data_train = f.readlines()
x0 = [''.join(line.strip().split(',')) for line in data_train]
x_train = []
for i in x0:
    x_train += [i.replace('\'', '').replace('[', '').replace(']', '')]
with open('val.csv') as f:
    data_val = f.readlines()
x1 = [''.join(line.strip().split(',')) for line in data_val]
x_val = []
for i in x1:
    x_val += [i.replace('\'', '').replace('[', '').replace(']', '')]
with open('test.csv') as f:
    data_test = f.readlines()
x2 = [''.join(line.strip().split(',')) for line in data_test]
x_test = []
for i in x2:
    x_test += [i.replace('\'', '').replace('[', '').replace(']', '')]
with open('label.csv') as f:
    data_label = f.readlines()


tokenizer.word_index
X_train = tokenizer.texts_to_sequences(x_train)
X_train = pad_sequences(X_train, maxlen=Max_Sent_len, padding='post')
X_val = tokenizer.texts_to_sequences(x_val)
X_val = pad_sequences(X_val, maxlen=Max_Sent_len, padding='post')
X_test = tokenizer.texts_to_sequences(x_test)
X_test = pad_sequences(X_test, maxlen=Max_Sent_len, padding='post')
y = data_label
labels = [int(label) for label in data_label]
y_train = labels[:len(x_train)]
y_val = labels[len(x_train): len(x_train) + len(x_val)]
y_test = labels[-len(x_test):]


w2v = gensim.models.KeyedVectors.load('w2v.model')
embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(tokenizer.word_index) + 1, EMBEDDING_dim))

for word, i in tokenizer.word_index.items():
    try:
        embeddings_vector = w2v[word]
    except KeyError:
        embeddings_vector = None
    if embeddings_vector is not None:
        embeddings_matrix[i] = embeddings_vector
del w2v


### Building Model

def build_model(act= 'relu'):
  from keras.layers import GRU
  model = Sequential()
  model.add(Embedding(len(embeddings_matrix),
                            output_dim=EMBEDDING_dim,
                            weights = [embeddings_matrix] , trainable=True, input_length=30, name='word_embedding_layer'))

  model.add(Dense (50, activation=act, activity_regularizer=regularizers.l2(0.01)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(2, activation='softmax'))
  model.summary()
  model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
  model.fit(X_train, y_train,
            batch_size=Batch_Size,
            epochs=N_Epochs, verbose = 1,
            validation_data=(X_test, y_test))
  score = model.evaluate(X_test, y_test, verbose=1)
  print("Test Score:", score[0])
  print("Test Accuracy:", score[1])
  return model


## Saving models :
model1 = build_model('relu')
model1.save('model_relu.hdf5')

model2 = build_model('sigmoid')
model2.save('model_sigmoid.hdf5')

model3 = build_model('tanh')
model3.save('model_tanh.hdf5')