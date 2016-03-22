import numpy as np
import cPickle
import os
from utils import load_glove_vec
from alphabet import Alphabet


def main():
    data_dir = "parsed_tweets"
    wemb_dir = 'embeddings/smiley_tweets_embedding_final'
    wemb_delimiter = ' '
    wemb_nidm = 52

    vocabs = [
        ('parsed_tweets/vocab_words.pickle','final'),
        ('parsed_tweets/vocab_hashtags.pickle','topn')
    ]
    for fname_vocab,name in vocabs:
        #get vocabulary
        alphabet = cPickle.load(open(fname_vocab))
        words = alphabet.keys()
        print "Vocab size", len(alphabet)

        #get embeddings
        fname,delimiter,ndim = (wemb_dir,wemb_delimiter,wemb_nidm)
        word2vec = load_glove_vec(fname,words,delimiter,ndim)

        ndim = len(word2vec[word2vec.keys()[0]])
        print 'ndim', ndim

        random_words_count = 0
        vocab_emb = np.zeros((len(alphabet) + 1, ndim),dtype='float32')
        rand_vec = np.random.uniform(-0.25, 0.25, ndim)

        for word,idx in alphabet.iteritems():
            word_vec = word2vec.get(word, None)
            if word_vec is None:
              word_vec = rand_vec
              random_words_count += 1
            vocab_emb[idx] = word_vec
        print "Using zero vector as random"
        print 'random_words_count', random_words_count
        print vocab_emb.shape
        outfile = os.path.join(data_dir, 'emb_smiley_tweets_embedding_{}.npy'.format(name))
        print outfile
        np.save(outfile, vocab_emb)




if __name__ == '__main__':
  main()
