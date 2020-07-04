

import sys
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


act = sys.argv[-1]
textfile_path = sys.argv[-2]
tokenizer = Tokenizer()
test = open(textfile_path).read().splitlines()
input = tokenizer.texts_to_sequences(test)
input = pad_sequences(input,maxlen=30,padding='post')
if act == 'relu':
    model = load_model('model_relu.hdf5')
    prediction = model.predict(input)
if act == 'sigmoid':
    model = load_model('model_sigmoid.hdf5')
    prediction = model.predict(input)
if act == 'tanh':
    model = load_model('model_tanh.hdf5')
    prediction = model.predict(input)

print(test,'is predicted to be',prediction)