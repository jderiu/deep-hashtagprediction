import sys
from nltk.tokenize import TweetTokenizer
from parse_tweets_sheffield import preprocess_tweet,convert_sentiment
from utils import load_glove_vec
import gzip
import cPickle
import os
import operator
import getopt


class Alphabet(dict):
    def __init__(self, start_feature_id=1):
        self.fid = start_feature_id
        self.first = start_feature_id

    def add(self, item):
        idx,freq = self.get(item, (None,None))
        if idx is None:
            idx = self.fid
            self[item] = (idx,1)
            self.fid += 1
        else:
            self[item] = (idx,freq + 1)
        return idx

    def dump(self, fname):
        with open(fname, "w") as out:
            for k in sorted(self.keys()):
                out.write("{}\t{}\n".format(k, self[k]))

    def purge_dict(self,input_fname,min_freq=5):
        #removes all words from the alphabet which occur less than 5 times or are not contained in the word embeddings
        emb_fname,delimiter,ndim = ('embeddings/smiley_tweets_embedding_final',' ',52)

        word2vec = load_glove_vec(emb_fname,{},delimiter,ndim)
        for k in self.keys():
            idx,freq = self[k]
            if freq < min_freq and word2vec.get(k, None) == None:
                del self[k]
            else:
                self[k] = idx

        #reset fid after deletion
        self['UNK'] = 0
        counter = self.first
        for k,idx in sorted(self.items(),key=operator.itemgetter(1)):
            self[k] = counter
            counter += 1
        self.fid = counter


def main():
    outdir = "semeval_parsed_200M"

    print outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    #supervised data
    train = "semeval/task-B-train-plus-dev.tsv"
    test = "semeval/task-B-test2014-twitter.tsv"
    dev = "semeval/twitter-test-gold-B.downloaded.tsv"
    test15 = "semeval/task-B-test2015-twitter.tsv"
    train16 = "semeval/task-A-train-2016.tsv"
    dev2016 = "semeval/task-A-dev-2016.tsv"
    devtest2016 = "semeval/task-A-devtest-2016.tsv"
    test2016 = "semeval/SemEval2016-task4-test.subtask-A.txt"

    #unsupervised data
    smiley_tweets_200M = 'semeval/smiley_tweets_200M.gz'

    alphabet = Alphabet(start_feature_id=0)
    alphabet.add('UNKNOWN_WORD_IDX')
    dummy_word_idx = alphabet.fid
    tknzr = TweetTokenizer(reduce_len=True)

    fnames = [
        (train,3),
        (dev,3),
        (test,3),
        (test15,3),
        (train16,2),
        (dev2016,2),
        (devtest2016,2),
        (test2016,2)
    ]

    fnames_gz = [smiley_tweets_200M]

    counter = 0

    for (fname,pos) in fnames:
        with open(fname,'r ') as f:
            for line in f:
                tweet = line.split('\t')[pos]
                tweet,_ = convert_sentiment(tweet)
                tweet = tknzr.tokenize(preprocess_tweet(tweet))
                for token in tweet:
                    alphabet.add(token)
        print len(alphabet)

    for fname in fnames_gz:
        with gzip.open(fname,'r') as f:
            for tweet in f:
                tweet,_ = convert_sentiment(tweet)
                tweet = tknzr.tokenize(preprocess_tweet(tweet))
                for token in tweet:
                    alphabet.add(token)
                counter += 1
                if (counter % 1000000) == 0:
                    print 'Precessed Tweets:',counter

        print len(alphabet)

    print 'Alphabet before purge:',len(alphabet)
    alphabet.purge_dict(input_fname=type,min_freq=10)
    print 'Alphabet after purge:',len(alphabet)
    cPickle.dump(alphabet, open(os.path.join(outdir, 'vocab.pickle'), 'w'))


if __name__ == '__main__':
    main()