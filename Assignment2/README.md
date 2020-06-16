# msci-text-analytics-s20
According to Originial Dataset which contains stopwords:

Unigram for Test Set: Accuracy Score is  0.8095898801264985
\nBigram for Test Set: Accuracy Score is  0.8236147048161898
\nBigram + Unigram for Test Set: Accuracy Score is  0.8342645716928538

According to Stopword removed Dataset:

Unigram for Test Set: Accuracy Score is  0.8005024937188285
Bigram for Test Set: Accuracy Score is  0.7269159135510807
Bigram + Unigram for Test Set: Accuracy Score is  0.805977425282184

Here are question asnwers:

1. Training without stopwords is working better here. 
I believe reducing words caused lack of information needed to develop a good training model.
especially in positive and negative recognition, there are stopwords which help model to build better estimation.
anIn addition, comments are written very informally, which has provided sentences that can not be recognized easily if we remove stopwords.
