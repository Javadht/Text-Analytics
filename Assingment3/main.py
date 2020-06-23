import re
import os
import sys
from gensim.models import Word2Vec


def main(data_path):
## Reading Dataset

    with open(os.path.join(data_path, 'pos.txt')) as f:
        postext = f.read()
    with open(os.path.join(data_path, 'neg.txt')) as f:
        negtext = f.read()
    tx = postext + negtext
    text = tx.lower()
## Importing prepared stopwords in order to change the raw data
    swopen = open("stopwords_en.txt", "r")
    swtext = swopen.read()

    ## Removing special characters from dataset
    sw = re.sub('[^.\n,\'A-Za-z0-9]+', ' ', swtext)
    #stop_words = sw.split()
    without_special_char = re.sub('[^\n\'A-Za-z0-9]+', ' ', text)
    splitted_text = without_special_char.split("\n")
    lines = [line.split() for line in splitted_text]
    w2v_model = Word2Vec(
        lines,
        size=100,
        window=5,
        min_count=1,
        workers=4,
    )
    w2v_model.save('w2v.model')



if __name__ == '__main__':
    main(sys.argv[1])
