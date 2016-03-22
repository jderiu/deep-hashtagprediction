from gensim import models
import gzip
from nltk import TweetTokenizer
import logging
import re


def preprocess_tweet(tweet):
    #lowercase and normalize urls
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',tweet)
    tweet = re.sub('@[^\s]+','<user>',tweet)
    try:
        tweet = tweet.decode('unicode_escape').encode('ascii','ignore')
    except:
        pass
    return tweet


class MySentences(object):
    def __init__(self, files):
        self.files = files
        self.tknzr = TweetTokenizer()

    def __iter__(self):
       for fname in self.files:
             for line in gzip.open(fname,'rb'):
                 tweet = preprocess_tweet(line)
                 tweet = self.tknzr.tokenize(tweet.decode('utf-8'))
                 yield filter(lambda word: ' ' not in word, tweet)


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    #unsupervised data
    hashtag_tweets = 'tweets/hashtag_tweets.gz'
    files = [hashtag_tweets]
    sentences = MySentences(files=files)
    model = models.Word2Vec(sentences, size=100, window=5, min_count=15, workers=8,sg=1,sample=1e-5,hs=1)
    model.save_word2vec_format('embeddings/hashtag_tweets_embedding',binary=False)


if __name__ == '__main__':
    main()