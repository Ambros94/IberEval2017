import preprocessor as p
from nltk import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer


def nothing(tweet):
    return tweet


def st(tweet):
    tokenizer = TweetTokenizer(reduce_len=True)
    word_list = tokenizer.tokenize(tweet)
    stemmer = SpanishStemmer()
    stemmed_words = [stemmer.stem(word=word) for word in word_list]
    tweet = " ".join([word for word in stemmed_words])
    return tweet


def st_sw(tweet):
    tokenizer = TweetTokenizer(reduce_len=True)
    word_list = tokenizer.tokenize(tweet)
    filtered_words = [word for word in word_list if word not in stopwords.words('spanish')]
    stemmer = SpanishStemmer()
    stemmed_words = [stemmer.stem(word=word) for word in filtered_words]
    tweet = " ".join([word for word in stemmed_words])
    return tweet


def st_sw_cl(tweet):
    p.set_options(p.OPT.URL, p.OPT.RESERVED)
    tweet = p.clean(tweet)
    tokenizer = TweetTokenizer(reduce_len=True)
    word_list = tokenizer.tokenize(tweet)
    filtered_words = [word for word in word_list if word not in stopwords.words('spanish')]
    stemmer = SpanishStemmer()
    stemmed_words = [stemmer.stem(word=word) for word in filtered_words]
    tweet = " ".join([word for word in stemmed_words])
    return tweet


def st_sw_cl_mt(tweet):
    p.set_options(p.OPT.URL, p.OPT.RESERVED)
    tweet = p.clean(tweet)
    p.set_options(p.OPT.MENTION)
    tweet = p.tokenize(tweet)
    tokenizer = TweetTokenizer(reduce_len=True)
    word_list = tokenizer.tokenize(tweet)
    filtered_words = [word for word in word_list if word not in stopwords.words('spanish')]
    stemmer = SpanishStemmer()
    stemmed_words = [stemmer.stem(word=word) for word in filtered_words]
    tweet = " ".join([word for word in stemmed_words])
    return tweet


def st_sw_cl_mt_num(tweet):
    p.set_options(p.OPT.URL, p.OPT.RESERVED)
    tweet = p.clean(tweet)
    p.set_options(p.OPT.MENTION, p.OPT.NUMBER)
    tweet = p.tokenize(tweet)
    tokenizer = TweetTokenizer(reduce_len=True)
    word_list = tokenizer.tokenize(tweet)
    filtered_words = [word for word in word_list if word not in stopwords.words('spanish')]
    stemmer = SpanishStemmer()
    stemmed_words = [stemmer.stem(word=word) for word in filtered_words]
    tweet = " ".join([word for word in stemmed_words])
    return tweet


def st_sw_cl_mt_num_em(tweet):
    p.set_options(p.OPT.URL, p.OPT.RESERVED)
    tweet = p.clean(tweet)
    p.set_options(p.OPT.MENTION, p.OPT.SMILEY, p.OPT.EMOJI, p.OPT.NUMBER)
    tweet = p.tokenize(tweet)
    tokenizer = TweetTokenizer(reduce_len=True)
    word_list = tokenizer.tokenize(tweet)
    filtered_words = [word for word in word_list if word not in stopwords.words('spanish')]
    stemmer = SpanishStemmer()
    stemmed_words = [stemmer.stem(word=word) for word in filtered_words]
    tweet = " ".join([word for word in stemmed_words])
    return tweet


def st_sw_cl_mt_num_em_ht(tweet):
    p.set_options(p.OPT.URL, p.OPT.RESERVED)
    tweet = p.clean(tweet)
    p.set_options(p.OPT.MENTION, p.OPT.SMILEY, p.OPT.EMOJI, p.OPT.HASHTAG, p.OPT.NUMBER)
    tweet = p.tokenize(tweet)
    tokenizer = TweetTokenizer(reduce_len=True)
    word_list = tokenizer.tokenize(tweet)
    filtered_words = [word for word in word_list if word not in stopwords.words('spanish')]
    stemmer = SpanishStemmer()
    stemmed_words = [stemmer.stem(word=word) for word in filtered_words]
    tweet = " ".join([word for word in stemmed_words])
    return tweet


def sw_cl_mt_num_em_ht(tweet):
    p.set_options(p.OPT.URL, p.OPT.RESERVED)
    tweet = p.clean(tweet)
    p.set_options(p.OPT.MENTION, p.OPT.SMILEY, p.OPT.EMOJI, p.OPT.NUMBER,p.OPT.HASHTAG)
    tweet = p.tokenize(tweet)
    tokenizer = TweetTokenizer(reduce_len=True)
    word_list = tokenizer.tokenize(tweet)
    filtered_words = [word for word in word_list if word not in stopwords.words('spanish')]
    tweet = " ".join([word for word in filtered_words])
    return tweet


def cl(tweet):
    p.set_options(p.OPT.URL, p.OPT.RESERVED)
    tweet = p.clean(tweet)
    return tweet


def cl_em(tweet):
    p.set_options(p.OPT.URL, p.OPT.RESERVED)
    tweet = p.clean(tweet)
    p.set_options(p.OPT.SMILEY, p.OPT.EMOJI)
    tweet = p.tokenize(tweet)
    return tweet


def cl_mt_num_em(tweet):
    p.set_options(p.OPT.URL, p.OPT.RESERVED, p.OPT.NUMBER)
    tweet = p.clean(tweet)
    p.set_options(p.OPT.MENTION, p.OPT.SMILEY, p.OPT.EMOJI)
    tweet = p.tokenize(tweet)
    return tweet


def clean_tweets(cleaning_function, tweets):
    cleaned_tweets = [cleaning_function(t) for t in tweets]
    return cleaned_tweets
