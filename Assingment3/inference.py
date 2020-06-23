import sys
from gensim.models import Word2Vec


def main(text):
    with open(text) as f:
        a = f.readlines()
    sample_text = [w.strip() for w in a]
    w2v = Word2Vec.load('w2v.model')
    for w in sample_text:
        word = w2v[w]
        top = w2v.wv.most_similar(positive=[word], topn=20)
        print(top)

if __name__ == '__main__':
    text = sys.argv[1]
    main(text)
