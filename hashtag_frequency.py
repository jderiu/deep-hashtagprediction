import gzip
import re
import cPickle
import os
import operator

p = re.compile('(#[^\s]+)')

def preprocess_tweet(tweet):
    #lowercase and normalize urls
    tweet = tweet.lower()
    #tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',tweet)
    #tweet = re.sub('@[^\s]+','<user>',tweet)

    hashtags = p.findall(tweet)
    try:
        tweet = tweet.decode('unicode_escape').encode('ascii','ignore')
    except:
        pass
    return tweet,hashtags


def create_htfreq_dict():
    data_dir = 'tweets/hashtag_smileys_tweets.gz'
    outdir = 'parsed_tweets'

    hashtag_freq = {}
    counter = 0
    with gzip.open(data_dir,'r') as f:
        for tweet in f:
            tweet,hashtags = preprocess_tweet(tweet)
            if len(hashtags) == 1:
                ht = hashtags[0]
                try:
                    hashtag_freq[ht] += 1
                except:
                    hashtag_freq[ht] = 1

            counter += 1
            if (counter%100000) == 0:
                print "Elements processed:",counter, 'Hashtags:',len(hashtag_freq)

    hashtag_freq = sorted(hashtag_freq.items(), key=operator.itemgetter(1))
    for i in xrange(0,100):
        print hashtag_freq[i]

    cPickle.dump(dict(hashtag_freq), open(os.path.join(outdir, 'hashtag_smiley_freq.pickle'), 'w'))


def filter_top_tweets(n=100):
    outdir = 'parsed_tweets'
    data_dir = 'tweets/hashtag_smileys_tweets.gz'

    hashtag_freq = cPickle.load(open(os.path.join(outdir, 'hashtag_smiley_freq.pickle')))
    hashtag_freq = sorted(hashtag_freq.items(), key=operator.itemgetter(1))
    hashtag_freq = reversed(hashtag_freq)

    top_n = []
    counter = 0
    for ht in hashtag_freq:
        top_n.append(ht[0])
        counter += 1
        if counter == n:
            break

    counter = 0
    ht_counter = 0
    f_out = gzip.open('tweets/hashtag_top100_smileys_tweets.gz','w')
    with gzip.open(data_dir,'r') as f:
        for line in f:
            tweet,hashtags = preprocess_tweet(line)
            if len(hashtags) == 1:
                    ht = hashtags[0]
                    if ht in top_n:
                        f_out.write(tweet)
                        ht_counter += 1

            counter += 1
            if (counter%1000000) == 0:
                print "Elements processed:",counter, 'Top tweets:',ht_counter

    f_out.flush()
    f_out.close()
    print ht_counter

def print_stats(n=100):
    outdir = 'parsed_tweets'

    hashtag_freq = cPickle.load(open(os.path.join(outdir, 'hashtag_smiley_freq.pickle')))
    hashtag_freq = sorted(hashtag_freq.items(), key=operator.itemgetter(1))
    hashtag_freq = reversed(hashtag_freq)

    top_n = []
    counter = 0
    sum = 0
    for ht in hashtag_freq:
        print ht
        sum += ht[1]
        counter += 1
        if counter == n:
            break
    print sum

if __name__ == '__main__':
    create_htfreq_dict()
    print_stats()
    filter_top_tweets()


