import numpy as np
import cPickle
import os
from utils import load_glove_vec
from alphabet import Alphabet


def main():
    HOME_DIR = "semeval_parsed"
    np.random.seed(123)
    input_fname = '200M'

    data_dir = HOME_DIR + '_' + input_fname

    #get vocabulary
    fname_vocab = os.path.join(data_dir, 'vocab.pickle')
    alphabet = cPickle.load(open(fname_vocab))
    words = alphabet.keys()
    print "Vocab size", len(alphabet)

    #get embeddings
    fname,delimiter,ndim = ('embeddings/smiley_tweets_embedding_final',' ',52)
    word2vec = load_glove_vec(fname,words,delimiter,ndim)

    ndim = len(word2vec[word2vec.keys()[0]])
    print 'ndim', ndim

    random_words_count = 0
    vocab_emb = np.zeros((len(alphabet) + 1, ndim),dtype='float32')
    for word,idx in alphabet.iteritems():
        word_vec = word2vec.get(word, None)
        if word_vec is None:
          word_vec = np.random.uniform(-0.25, 0.25, ndim)
          random_words_count += 1
        vocab_emb[idx] = word_vec
    print "Using zero vector as random"
    print 'random_words_count', random_words_count
    print vocab_emb.shape
    outfile = os.path.join(data_dir, 'emb_smiley_tweets_embedding_final.npy')
    print outfile
    np.save(outfile, vocab_emb)

if __name__ == '__main__':
  main()
