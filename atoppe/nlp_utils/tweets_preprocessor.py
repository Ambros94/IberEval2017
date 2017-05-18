import preprocessor as p


def _clean_tweet(tweet):
    p.set_options(p.OPT.URL, p.OPT.RESERVED)
    tweet = p.clean(tweet)
    p.set_options(p.OPT.EMOJI, p.OPT.SMILEY)
    tweet = p.tokenize(tweet)

    return tweet


def clean_tweets(tweets):
    cleaned_tweets = [_clean_tweet(t) for t in tweets]
    return cleaned_tweets
