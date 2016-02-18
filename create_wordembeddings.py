from gensim import models
import gzip
import sys
from parse_tweets_sheffield import preprocess_tweet
from nltk import TweetTokenizer
import logging


class MySentences(object):
    def __init__(self, files):
        self.files = files
        self.tknzr = TweetTokenizer()

    def __iter__(self):
       for (fname,pos) in self.files:
             for line in gzip.open(fname,'rb'):
                 tweet = line.split('\t')[pos]
                 tweet = preprocess_tweet(tweet)
                 tweet = self.tknzr.tokenize(tweet.decode('utf-8'))
                 yield filter(lambda word: ' ' not in word, tweet)


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    input_fname = ''
    if len(sys.argv) > 1:
        input_fname = sys.argv[1]

    #supervised data
    train = "semeval/task-B-train-plus-dev.tsv.gz"
    dev = "semeval/twitter-test-gold-B.downloaded.tsv.gz"
    train16 = "semeval/task-A-train-2016.tsv.gz"
    dev2016 = "semeval/task-A-dev-2016.tsv.gz"
    devtest2016 = "semeval/task-A-devtest-2016.tsv.gz"
    test2016 = "semeval/SemEval2016-task4-test.subtask-A.txt.gz"

    #unsupervised data
    smiley_pos = 'semeval/smiley_tweets_{}.gz'.format(input_fname)

    files = [(train,3),
             (dev,3),
             (train16,2),
             (dev2016,2),
             (devtest2016,2),
             (test2016,2),
             (smiley_pos,0)]
    sentences = MySentences(files=files)
    model = models.Word2Vec(sentences, size=52, window=5, min_count=5, workers=7,sg=1,sample=1e-5,hs=1)
    model.save_word2vec_format('embeddings/smiley_tweets_embedding_final{}'.format(input_fname),binary=False)


if __name__ == '__main__':
    main()