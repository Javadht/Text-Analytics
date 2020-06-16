import re
import random

## Reading Dataset
pos = open("pos.txt", 'r')
postext = pos.read()
neg = open("neg.txt", 'r')
negtext = neg.read()
tx = postext + negtext
text = tx.lower()

## Importing prepared stopwords in order to change the raw data
swopen = open("stopwords_en.txt", "r")
swtext = swopen.read()

## Removing special characters from dataset
sw = re.sub('[^.\n,\'A-Za-z0-9]+', ' ', swtext)
stop_words = sw.split()
without_special_char = re.sub('[^\n\'A-Za-z0-9]+', ' ', text)
splitted_text = without_special_char.split("\n")

## Making label for each row
labelized_text = list(zip(splitted_text, [1]*len(postext.split("\n")) + [0]*len(negtext.split("\n"))))

## Making a copy for randomization
t = labelized_text[:]
random.shuffle(t)

## Tokenization with stopwords
tokenized = []
label = []
for i in t:
    tokenized += [i[0].split()]
    label += [i[1]]

## Tokenization without stopwords
tokenized_sw = []
for line in tokenized:
    t2 = [j for j in line if j not in stop_words]
    tokenized_sw.append(t2)

## Tokenization without stopwords for randomized dataset
tokenized_without_sw = []
for line in t:
    t2 = [j for j in line if j not in stop_words]
    tokenized_without_sw.append(t2)

## Splitting tokenized corpus into 3 different parts; Train, Validation, Test.
training = tokenized[:int(len(tokenized)*0.8)]
validation = tokenized[int(len(tokenized)*0.8):int(len(tokenized)*0.9)]
testing = tokenized[int(len(tokenized)*0.9):]

## Splitting tokenized corpus without stopwords into 3 different parts; Train, Validation, Test.
training_without_sw = tokenized_sw[:int(len(tokenized_sw)*0.8)]
validation_without_sw = tokenized_sw[int(len(tokenized_sw)*0.8):int(len(tokenized_sw)*0.9)]
testing_without_sw = tokenized_sw[int(len(tokenized_sw)*0.9):]

## Making CSV of each file
output = "\n".join(str(e) for e in tokenized)
with open('output.csv', 'w', newline='') as f:
    f.write(output)
output_sw = "\n".join(str(e) for e in tokenized_sw)
with open('output_sw.csv', 'w', newline='') as f:
    f.write(output_sw)
train = "\n".join(str(e) for e in training)
with open('train.csv', 'w', newline='') as f:
    f.write(train)
train_sw = "\n".join(str(e) for e in training_without_sw)
with open('train_sw.csv', 'w', newline='') as f:
    f.write(train_sw)
val = "\n".join(str(e) for e in validation)
with open('val.csv', 'w', newline='') as f:
    f.write(val)
val_sw = "\n".join(str(e) for e in validation_without_sw)
with open('val_sw.csv', 'w', newline='') as f:
    f.write(val_sw)
test = "\n".join(str(e) for e in testing)
with open('test.csv', 'w', newline='') as f:
    f.write(test)
test_sw = "\n".join(str(e) for e in testing_without_sw)
with open('test_sw.csv', 'w', newline='') as f:
    f.write(test_sw)
labels = "\n".join(str(e) for e in label)
with open('label.csv', 'w', newline='') as f:
    f.write(labels)