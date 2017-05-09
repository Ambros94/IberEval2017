import preprocessor as p
from nltk import TweetTokenizer
from nltk.corpus import stopwords


def _clean_tweet(tweet):
    p.set_options(p.OPT.URL, p.OPT.RESERVED, p.OPT.NUMBER)
    tweet = p.clean(tweet)
    p.set_options(p.OPT.MENTION, p.OPT.SMILEY, p.OPT.EMOJI)
    tweet = p.tokenize(tweet)
    tokenizer = TweetTokenizer(reduce_len=True)
    word_list = tokenizer.tokenize(tweet)
    filtered_words = [word for word in word_list if word not in stopwords.words('spanish')]
    # stemmer = SpanishStemmer()
    # stemmed_words = [stemmer.stem(word=word) for word in filtered_words]
    tweet = " ".join([word for word in filtered_words])
    return tweet


def clean_tweets(tweets):
    cleaned_tweets = [_clean_tweet(t) for t in tweets]
    return cleaned_tweets
