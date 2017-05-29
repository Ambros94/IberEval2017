import preprocessor as p
from nltk import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer


def kim_coset(tweet):
    """
    Stopwords removal, Reserved words removal, substitute numbers, substitute emoji
    :param tweet: 
    :return: 
    """
    p.set_options(p.OPT.RESERVED)
    tweet = p.clean(tweet)
    p.set_options(p.OPT.EMOJI, p.OPT.SMILEY)
    tweet = p.tokenize(tweet)
    tokenizer = TweetTokenizer(reduce_len=True)
    word_list = tokenizer.tokenize(tweet)
    filtered_words = [word for word in word_list if word not in stopwords.words('spanish')]
    tweet = " ".join([word for word in filtered_words])
    return tweet


def cnn_gender_es(tweet):
    """
    Remove URLs, Substitute, emoji
    :param tweet:
    :return:
    """
    p.set_options(p.OPT.URL)
    tweet = p.clean(tweet)
    p.set_options(p.OPT.EMOJI)
    return p.tokenize(tweet)


def cnn_gender_ca(tweet):
    """
    Remove URLs, reserved words, Substitute hashtags, numbers, emoji
    :param tweet:
    :return:
    """
    p.set_options(p.OPT.URL, p.OPT.RESERVED)
    tweet = p.clean(tweet)
    p.set_options(p.OPT.HASHTAG, p.OPT.NUMBER, p.OPT.EMOJI)
    return p.tokenize(tweet)


def fast_text_coset(tweet):
    """
    Stopwords removal, Reserved words removal, substitute numbers, substitute emoji
    :param tweet: 
    :return: 
    """
    p.set_options(p.OPT.URL)
    tweet = p.clean(tweet)
    p.set_options(p.OPT.MENTION, p.OPT.NUMBER)
    tweet = p.tokenize(tweet)
    tokenizer = TweetTokenizer(reduce_len=True)
    word_list = tokenizer.tokenize(tweet)
    filtered_words = [word for word in word_list if word not in stopwords.words('spanish')]
    tweet = " ".join([word for word in filtered_words])
    return tweet


def nothing(tweet):
    """
    Returns the tweet itself, used as baseline to compare other models
    :param tweet: 
    :return: 
    """
    return tweet


def stemming(tweet):
    """
    Performing stemming using NLTK spanish Stemmer and TweetTokenizer
    :param tweet: 
    :return: 
    """
    tokenizer = TweetTokenizer(reduce_len=True)
    word_list = tokenizer.tokenize(tweet)
    stemmer = SpanishStemmer()
    stemmed_words = [stemmer.stem(word=word) for word in word_list]
    tweet = " ".join([word for word in stemmed_words])
    return tweet


def stop_words(tweet):
    """
    Removed stop words using TweetTokenizer and NLTK list of spanish stop words
    :param tweet: 
    :return: 
    """
    tokenizer = TweetTokenizer(reduce_len=True)
    word_list = tokenizer.tokenize(tweet)
    filtered_words = [word for word in word_list if word not in stopwords.words('spanish')]
    tweet = " ".join([word for word in filtered_words])
    return tweet


def remove_url(tweet):
    """
    Removed all the occurrences of URLs in the sentence
    :param tweet: 
    :return: 
    """
    p.set_options(p.OPT.URL)
    return p.clean(tweet)


def removed_reserved_words(tweet):
    """
    Remove all the occurrences of reserved words (e.g. R.T.)
    :param tweet: 
    :return: 
    """
    p.set_options(p.OPT.RESERVED)
    return p.clean(tweet)


def sub_mentions(tweet):
    """
    Substitute all the occurrences of mentions (e.g. @Ambros) with $MENTION
    :param tweet: 
    :return: 
    """
    p.set_options(p.OPT.MENTION)
    return p.tokenize(tweet)


def sub_emoji(tweet):
    """
    Substitute all the occurrences of emojies (e.g. :)-> unicode char) with $EMOJI
    :param tweet: 
    :return: 
    """
    p.set_options(p.OPT.EMOJI)
    return p.tokenize(tweet)


def sub_smiley(tweet):
    """
    Substitute all the occurrences of smiles (e.g. :)-> two different chars) with $SMILEY
    :param tweet: 
    :return: 
    """
    p.set_options(p.OPT.SMILEY)
    return p.tokenize(tweet)


def sub_hashtags(tweet):
    """
    Substitute all the occurrences of hashtags (e.g. #atoppe) with $HASHTAG
    :param tweet: 
    :return: 
    """
    p.set_options(p.OPT.MENTION)
    return p.tokenize(tweet)


def sub_numbers(tweet):
    """
    Substitute all the occurrences of numbers (e.g. 17.45) with $NUMBER
    :param tweet: 
    :return: 
    """
    p.set_options(p.OPT.NUMBER)
    return p.tokenize(tweet)


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
    p.set_options(p.OPT.MENTION, p.OPT.SMILEY, p.OPT.EMOJI, p.OPT.NUMBER, p.OPT.HASHTAG)
    tweet = p.tokenize(tweet)
    tokenizer = TweetTokenizer(reduce_len=True)
    word_list = tokenizer.tokenize(tweet)
    filtered_words = [word for word in word_list if word not in stopwords.words('spanish')]
    tweet = " ".join([word for word in filtered_words])
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
