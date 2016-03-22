import numpy as np
from nltk.tokenize import TweetTokenizer
import gzip
from alphabet import Alphabet
import re
import cPickle
import os
from collections import OrderedDict


def preprocess_tweet(tweet):
    #lowercase and normalize urls
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',tweet)
    tweet = re.sub('@[^\s]+','<user>',tweet)

    p = re.compile('(#[^\s]+)')
    hashtags = p.findall(tweet)
    try:
        tweet = tweet.decode('unicode_escape').encode('ascii','ignore')
    except:
        pass
    return tweet,hashtags


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


def store_file(f_in, f_out, alphabet_words,alphabet_hashtags, dummy_word_idx, hashtag_fname=None):
    #stores the tweets in batches so it fits in memory
    tknzr = TweetTokenizer(reduce_len=True)
    counter = 0
    batch_counter = 0
    output = open(f_out,'wb')
    output_hashtag = open(hashtag_fname, 'wb')
    batch_size = 500000
    tweet_batch = []
    hashtag_batch=[]
    with gzip.open(f_in,'r') as f:
        for tweet in f:
            tweet,hashtags = preprocess_tweet(tweet)
            if len(hashtags) == 1:
                ht = hashtags[0]
                alphabet_hashtags.add(ht)
                ht_idx = alphabet_hashtags.get(ht,UNKNOWN_HASHTAG_IDX)

                tweet = tweet.replace(ht,'')
                tweet_tok = tknzr.tokenize(tweet.decode('utf-8'))
                tweet_batch.append(tweet_tok)
                hashtag_batch.append(ht_idx)

                batch_counter += 1

                for token in tweet_tok:
                    alphabet_words.add(token)

                if batch_counter >= batch_size:
                    tweet_idx = convert2indices(tweet_batch, alphabet_words, dummy_word_idx)
                    np.save(output,tweet_idx)
                    np.save(output_hashtag,hashtag_batch)
                    print 'Saved tweets:',tweet_idx.shape
                    tweet_batch = []
                    hashtag_batch=[]
                    batch_counter = 0
                counter += 1
                if (counter%1000000) == 0:
                    print "Elements processed:",counter

    tweet_idx = convert2indices(tweet_batch, alphabet_words, dummy_word_idx)
    np.save(output,tweet_idx)
    np.save(output_hashtag,hashtag_batch)
    print len(alphabet_hashtags)
    print len(alphabet_words)
    print 'Saved tweets:',tweet_idx.shape
    return counter

UNKNOWN_WORD_IDX = 0
UNKNOWN_HASHTAG_IDX = 0
DUMMY_WORD_IDX = 1

def main():
    data_dir = 'tweets/hashtag_top100_smileys_tweets_{}.gz'
    output_dir_tweets = 'parsed_tweets/hashtag_top100_smiley_tweets_{}.tweets.npy'
    output_dir_hashtags = 'parsed_tweets/hashtag_top100_smiley_tweets_{}.hashtags.npy'
    outdir = 'parsed_tweets'

    alphabet_words = Alphabet(start_feature_id=0)
    alphabet_words.add('UNKNOWN_WORD_IDX')
    alphabet_words.add('DUMMY_WORD_IDX')
    dummy_word_idx = DUMMY_WORD_IDX

    alphabet_hashtags = Alphabet(start_feature_id=0)
    alphabet_hashtags.add('UNKNOWN_HASHTAG_IDX')

    inp = 'train'
    store_file(data_dir.format(inp),output_dir_tweets.format(inp),alphabet_words,alphabet_hashtags,dummy_word_idx,output_dir_hashtags.format(inp))
    inp = 'test'
    store_file(data_dir.format(inp),output_dir_tweets.format(inp),alphabet_words,alphabet_hashtags,dummy_word_idx,output_dir_hashtags.format(inp))

    cPickle.dump(alphabet_words, open(os.path.join(outdir, 'vocab_words.pickle'), 'w'))
    cPickle.dump(alphabet_hashtags, open(os.path.join(outdir, 'vocab_hashtags.pickle'), 'w'))

if __name__ == '__main__':
    main()
