# Assignment3 Report

Accordint to the results, we have following irrelevant adjectives:
good  | bad
------------- | -------------
('bad', 0.6680839657783508)  | ('good', 0.6680839657783508)
('poor', 0.6061838269233704)  | ('funny', 0.5950542688369751) 

Therefore, we can conclude it is not the case.
The reason behind this is that vectorization of word, can not understand deep meaning of the word itself. it just considers probabilities of the happening words and position of the word against other similar words.
Another issue is that some adjectives can be used both in negative and positive context, which w2v is an unreliable way to distinguish them.
