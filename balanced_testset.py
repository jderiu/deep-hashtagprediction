import gzip
import re
import cPickle
import os
import operator


p = re.compile('(#[^\s]+)')

def preprocess_tweet(tweet):
    #lowercase and normalize urls
    tweet = tweet.lower()
    hashtags = p.findall(tweet)
    try:
        tweet = tweet.decode('unicode_escape').encode('ascii','ignore')
    except:
        pass
    return tweet,hashtags



def main():
    data_dir = 'tweets/hashtag_top100_smileys_tweets.gz'
    outdir = 'parsed_tweets'
    n = 100

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

    f_out_train = gzip.open('tweets/hashtag_top100_smileys_tweets_train.gz','w')
    f_out_test = gzip.open('tweets/hashtag_top100_smileys_tweets_test.gz','w')

    balancer_dict = {}
    for ht in top_n:
        balancer_dict[ht] = 0

    print len(balancer_dict)
    print len(top_n)
    counter = 0
    train_counter = 0
    test_counter = 0
    with gzip.open(data_dir,'r') as f:
        for line in f:
            tweet,hashtags = preprocess_tweet(line)
            if len(hashtags) == 1:
                ht = hashtags[0]
                b = balancer_dict.get(ht,None)
                if b is not None and b < 1000:
                    f_out_test.write(tweet)
                    balancer_dict[ht] += 1
                    test_counter += 1
                else:
                    train_counter += 1
                    f_out_train.write(tweet)
                counter += 1
                if (counter%500000) == 0:
                    print "Elements processed:",counter, 'Train Tweets:',train_counter,'Test Tweets:',test_counter
            else:
                print 'Len not 1:',len(hashtags)

    print balancer_dict
    print len(balancer_dict)
    print train_counter + test_counter
    print test_counter
    print counter
    f_out_train.close()
    f_out_test.close()

if __name__ == '__main__':
    main()