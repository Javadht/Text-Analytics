from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
## Reading Data : I should Mention that because my first assignment output had something like qoutation and brackets, I had to remove them here:
with open('train.csv') as f:
    data_train = f.readlines()
x0 = [''.join(line.strip().split(',')) for line in data_train]
x_train = []
for i in x0:
    x_train += [i.replace('\'', '').replace('[', '').replace(']', '')]

with open('train_sw.csv') as f:
    data_train_ns = f.readlines()
x00 = [''.join(line.strip().split(',')) for line in data_train]
x_train_ns = []
for i in x00:
    x_train_ns += [i.replace('\'', '').replace('[', '').replace(']', '')]

with open('val.csv') as f:
    data_val = f.readlines()
x1 = [''.join(line.strip().split(',')) for line in data_val]
x_val = []
for i in x1:
    x_val += [i.replace('\'', '').replace('[', '').replace(']', '')]

with open('val_sw.csv') as f:
    data_val_ns = f.readlines()
x10 = [''.join(line.strip().split(',')) for line in data_val_ns]
x_val_ns = []
for i in x10:
    x_val_ns += [i.replace('\'', '').replace('[', '').replace(']', '')]

with open('test.csv') as f:
    data_test = f.readlines()
x2 = [''.join(line.strip().split(',')) for line in data_test]
x_test = []
for i in x2:
    x_test += [i.replace('\'', '').replace('[', '').replace(']', '')]

with open('test_sw.csv') as f:
    data_test_ns = f.readlines()
x20 = [''.join(line.strip().split(',')) for line in data_test_ns]
x_test_ns = []
for i in x20:
    x_test_ns += [i.replace('\'', '').replace('[', '').replace(']', '')]

with open('label.csv') as f:
    data_label = f.readlines()

## Making lists of input data
labels = [int(label) for label in data_label]
y_train = labels[:len(x_train)]
y_val = labels[len(x_train): len(x_train) + len(x_val)]
y_test = labels[-len(x_test):]

## 2. Training for dataset with stopwords
    ## 2.1 Unigram
uni_count_vector= CountVectorizer(ngram_range=(1,1))
uni_x_train_count = uni_count_vector.fit_transform(x_train)
tfidf_transformer = TfidfTransformer()
uni_x_train_tfidf = tfidf_transformer.fit_transform(uni_x_train_count)
uni_clf = MultinomialNB().fit(uni_x_train_tfidf, y_train)
uni_x_test = uni_count_vector.transform(x_test)
uni_x_test_tfidf = tfidf_transformer.transform(uni_x_test)
uni_x_val = uni_count_vector.transform(x_val)
uni_x_val_tfidf = tfidf_transformer.transform(uni_x_val)
pred_uni_x_val = uni_clf.predict(uni_x_val_tfidf)
pred_uni_x_test = uni_clf.predict(uni_x_test_tfidf)
print('Unigram for Valid Set: Accuracy Score is',accuracy_score(y_val,pred_uni_x_val))
print('Unigram for Test Set: Accuracy Score is ',accuracy_score(y_test,pred_uni_x_test))
with open("mnb_uni.pkl", 'wb') as f:
    pickle.dump(uni_clf, f)

    ## 2.2 Bigram
bi_count_vector= CountVectorizer(ngram_range=(2,2))
bi_x_train_count = bi_count_vector.fit_transform(x_train)
tfidf_transformer = TfidfTransformer()
bi_x_train_tfidf = tfidf_transformer.fit_transform(bi_x_train_count)
bi_clf = MultinomialNB().fit(bi_x_train_tfidf, y_train)
bi_x_test = bi_count_vector.transform(x_test)
bi_x_test_tfidf = tfidf_transformer.transform(bi_x_test)
bi_x_val = bi_count_vector.transform(x_val)
bi_x_val_tfidf = tfidf_transformer.transform(bi_x_val)
pred_bi_x_val = bi_clf.predict(bi_x_val_tfidf)
pred_bi_x_test = bi_clf.predict(bi_x_test_tfidf)
print('Bigram for Valid Set: Accuracy Score is',accuracy_score(y_val,pred_bi_x_val))
print('Bigram for Test Set: Accuracy Score is ',accuracy_score(y_test,pred_bi_x_test))
with open("mnb_bi.pkl", 'wb') as f:
    pickle.dump(bi_clf, f)

    ## 2.3 Bigram + Unigram
both_count_vector= CountVectorizer(ngram_range=(1,2))
both_x_train_count = both_count_vector.fit_transform(x_train)
tfidf_transformer = TfidfTransformer()
both_x_train_tfidf = tfidf_transformer.fit_transform(both_x_train_count)
both_clf = MultinomialNB().fit(both_x_train_tfidf, y_train)
both_x_test = both_count_vector.transform(x_test)
both_x_test_tfidf = tfidf_transformer.transform(both_x_test)
both_x_val = both_count_vector.transform(x_val)
both_x_val_tfidf = tfidf_transformer.transform(both_x_val)
pred_both_x_val = both_clf.predict(both_x_val_tfidf)
pred_both_x_test = both_clf.predict(both_x_test_tfidf)
print('Bigram + Unigram for Valid Set: Accuracy Score is',accuracy_score(y_val,pred_both_x_val))
print('Bigram + Unigram for Test Set: Accuracy Score is ',accuracy_score(y_test,pred_both_x_test))
with open("mnb_uni_bi.pkl", 'wb') as f:
    pickle.dump(both_clf, f)

## 3. Training for dataset without stopwords
    ## 3.1 Unigram
ns_uni_count_vector= CountVectorizer(ngram_range=(1,1))
ns_uni_x_train_count = ns_uni_count_vector.fit_transform(x_train_ns)
tfidf_transformer = TfidfTransformer()
ns_uni_x_train_tfidf = tfidf_transformer.fit_transform(ns_uni_x_train_count)
ns_uni_clf = MultinomialNB().fit(ns_uni_x_train_tfidf, y_train)
ns_uni_x_test = ns_uni_count_vector.transform(x_test_ns)
ns_uni_x_test_tfidf = tfidf_transformer.transform(ns_uni_x_test)
ns_uni_x_val = ns_uni_count_vector.transform(x_val_ns)
ns_uni_x_val_tfidf = tfidf_transformer.transform(ns_uni_x_val)
pred_ns_uni_x_val = ns_uni_clf.predict(ns_uni_x_val_tfidf)
pred_ns_uni_x_test = ns_uni_clf.predict(ns_uni_x_test_tfidf)
print('Unigram for Valid Set: Accuracy Score is',accuracy_score(y_val,pred_ns_uni_x_val))
print('Unigram for Test Set: Accuracy Score is ',accuracy_score(y_test,pred_ns_uni_x_test))
with open("mnb_uni_ns.pkl", 'wb') as f:
    pickle.dump(ns_uni_clf, f)

    ## 3.2 Bigram
ns_bi_count_vector= CountVectorizer(ngram_range=(2,2))
ns_bi_x_train_count = ns_bi_count_vector.fit_transform(x_train_ns)
tfidf_transformer = TfidfTransformer()
ns_bi_x_train_tfidf = tfidf_transformer.fit_transform(ns_bi_x_train_count)
ns_bi_clf = MultinomialNB().fit(ns_bi_x_train_tfidf, y_train)
ns_bi_x_test = ns_bi_count_vector.transform(x_test_ns)
ns_bi_x_test_tfidf = tfidf_transformer.transform(ns_bi_x_test)
ns_bi_x_val = ns_bi_count_vector.transform(x_val_ns)
ns_bi_x_val_tfidf = tfidf_transformer.transform(ns_bi_x_val)
pred_ns_bi_x_val = ns_bi_clf.predict(ns_bi_x_val_tfidf)
pred_ns_bi_x_test = ns_bi_clf.predict(ns_bi_x_test_tfidf)
print('Bigram for Valid Set: Accuracy Score is',accuracy_score(y_val,pred_ns_bi_x_val))
print('Bigram for Test Set: Accuracy Score is ',accuracy_score(y_test,pred_ns_bi_x_test))
with open("mnb_bi_ns.pkl", 'wb') as f:
    pickle.dump(ns_bi_clf, f)

    ## 3.3 Bigram + Unigram
ns_both_count_vector= CountVectorizer(ngram_range=(1,2))
ns_both_x_train_count = ns_both_count_vector.fit_transform(x_train_ns)
tfidf_transformer = TfidfTransformer()
ns_both_x_train_tfidf = tfidf_transformer.fit_transform(ns_both_x_train_count)
ns_both_clf = MultinomialNB().fit(ns_both_x_train_tfidf, y_train)
ns_both_x_test = ns_both_count_vector.transform(x_test_ns)
ns_both_x_test_tfidf = tfidf_transformer.transform(ns_both_x_test)
ns_both_x_val = ns_both_count_vector.transform(x_val_ns)
ns_both_x_val_tfidf = tfidf_transformer.transform(ns_both_x_val)
pred_ns_both_x_val = ns_both_clf.predict(ns_both_x_val_tfidf)
pred_ns_both_x_test = ns_both_clf.predict(ns_both_x_test_tfidf)
print('Bigram + Unigram for Valid Set: Accuracy Score is',accuracy_score(y_val,pred_ns_both_x_val))
print('Bigram + Unigram for Test Set: Accuracy Score is ',accuracy_score(y_test,pred_ns_both_x_test))
with open("mnb_uni_bi_ns.pkl", 'wb') as f:
    pickle.dump(ns_both_clf, f)