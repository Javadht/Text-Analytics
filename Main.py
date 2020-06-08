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



## Tokenization with stopwords
tokenized = []
for i in splitted_text:
    tokenized += [i.split()]
## Making a copy for randomization
t = tokenized[:]
random.shuffle(t)

## Tokenization without stopwords
tokenized_without_sw = []
for line in t:
    t2 = [j for j in line if j not in stop_words]
    tokenized_without_sw.append(t2)


# print(len(tokenized_without_sw))
# # print(tokenized)
## Splitting tokenized corpus into 3 different parts; Train, Validation, Test.

training = t[:int(len(t)*0.8)]
validation = t[int(len(t)*0.8):int(len(t)*0.9)]
testing = t[int(len(t)*0.9):]

# random.shuffle(tokenized_without_sw)
## Splitting tokenized corpus without stopwords into 3 different parts; Train, Validation, Test.

training_without_sw = tokenized_without_sw[:int(len(tokenized_without_sw)*0.8)]
validation_without_sw = tokenized_without_sw[int(len(tokenized_without_sw)*0.8):int(len(tokenized_without_sw)*0.9)]
testing_without_sw = tokenized_without_sw[int(len(tokenized_without_sw)*0.9):]

# print(len(training))
# print(validation)
# print(testing)
## Making CSV of each file
output = "\n".join(str(e) for e in tokenized)
with open('output.csv', 'w', newline='') as f:
    f.write(output)
output_sw = "\n".join(str(e) for e in tokenized_without_sw)
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