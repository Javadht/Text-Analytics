# msci-text-analytics-s20
## Assignemnt2
### Result Report
In the following table you can see Accuracy Scores, from different built models:
Containing Stopwords  | Without Stopwords
------------- | -------------
Unigram: 0.8095898801264985  | Unigram: 0.8005024937188285 
Bigram: 0.8236147048161898  | Bigram: 0.7269159135510807
Bigram + Unigram: 0.8342645716928538  | Bigram + Unigram: 0.805977425282184

Here are question asnwers:

1. Training without stopwords is working better here. 
I believe reducing words caused lack of information needed to develop a good training model.
In positive and negative comment recognition, there are stopwords which may help model to build better estimation.
In addition, comments are written informally, which is composed of sentences that can not be recognized easily if we remove stopwords.
People try to convey their messages in the fastest way, therefor, removing words from already brief written senteces is not helpful.

2. Training with Bigram+Unigram is better: that's because model can understand conditional probabilities in different combinations both individually and together. 
it offers not only the power of two words sitting next to each other, but also considers each word individually, which recovers some combinations that may have conflict when two words are considered together. Additionally, lots of verbs are used in English that should be understood within other words(turn you homework...). therefore, the more combinations, the better training.
