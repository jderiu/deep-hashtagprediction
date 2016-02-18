import numpy as np
from nltk.tokenize import TweetTokenizer
import gzip
import re
import operator


def preprocess_tweet(tweet):
    #lowercase and normalize urls
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',tweet)
    #tweet = re.sub('@[^\s]+','<user>',tweet)
    #tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    try:
        tweet = tweet.decode('unicode_escape').encode('ascii','ignore')
    except:
        pass
    return tweet

emo_dict = {}


def read_emo(path):
    #read the smiley-score from the file
    with open(path) as f:
        counter = 0
        for line in f:
            splits = line.split(" ")
            emo_dict[splits[0].decode('unicode-escape').encode('latin-1').decode('utf-8')] = float(splits[4])
            counter += 1
    print emo_dict
    print max(emo_dict.iteritems(), key=operator.itemgetter(1))[0]


def convert_sentiment(tweet,trim=True):
    #given a tweet infer the sentiment by taking the label of the smiley-type which occurs most often
    tweet = tweet.decode('utf-8','ignore')
    emos = {}
    for emo,score in emo_dict.iteritems():
        emo_count = tweet.count(emo)
        if emo_count > 0:
            emos[emo] = emo_count
            if trim:
                tweet = tweet.replace(emo,"")
    if emos:
        max_emo = max(emos.iteritems(), key=operator.itemgetter(1))[0]
        sentiment = emo_dict[max_emo]
    else:
        sentiment = -np.inf

    return tweet.encode('utf-8'), sentiment

UNKNOWN_WORD_IDX = 0


def convert2indices(data, alphabet, dummy_word_idx, max_sent_length=140):
  data_idx = []
  max_len = 0
  unknown_words = 0
  for sentence in data:
    ex = np.ones(max_sent_length) * dummy_word_idx
    max_len = max(len(sentence),max_len)
    if len(sentence) > max_sent_length:
        print "Sentence length:",len(sentence)
        print sentence
    for i, token in enumerate(sentence):
      idx = alphabet.get(token, UNKNOWN_WORD_IDX)
      ex[i] = idx
      if idx == UNKNOWN_WORD_IDX:
          unknown_words += 1
    data_idx.append(ex)
  data_idx = np.array(data_idx).astype('int32')
  print "Max length in this batch:",max_len
  print "Number of unknown words:",unknown_words
  return data_idx


def store_file(f_in,f_out,alphabet,dummy_word_idx,sentiment_fname=None):
    #stores the tweets in batches so it fits in memory
    tknzr = TweetTokenizer(reduce_len=True)
    counter = 0
    output = open(f_out,'wb')
    if sentiment_fname:
        print 'Create sentiments',sentiment_fname
        output_sentiment = open(sentiment_fname,'wb')
    batch_size = 500000
    tweet_batch = []
    sentiment_batch=[]
    read_emo('emoscores')
    with gzip.open(f_in,'r') as f:
        for tweet in f:
            tweet, sentiment = convert_sentiment(tweet)
            if sentiment_fname and sentiment == -np.inf:
                continue
            tweet = preprocess_tweet(tweet)
            tweet = tknzr.tokenize(tweet.decode('utf-8'))
            tweet_batch.append(tweet)
            sentiment_batch.append(sentiment)
            counter += 1
            if counter%batch_size == 0:
                tweet_idx = convert2indices(tweet_batch,alphabet,dummy_word_idx)
                np.save(output,tweet_idx)
                if sentiment_fname:
                    np.save(output_sentiment,sentiment_batch)
                print 'Saved tweets:',tweet_idx.shape
                tweet_batch = []
                sentiment_batch=[]
            if (counter%1000000) == 0:
                print "Elements processed:",counter
    tweet_idx = convert2indices(tweet_batch,alphabet,dummy_word_idx)
    np.save(output,tweet_idx)
    np.save(output_sentiment,sentiment_batch)
    print 'Saved tweets:',tweet_idx.shape
    return counter


