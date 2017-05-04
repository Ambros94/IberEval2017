import csv
import os

import preprocessor as p
import sklearn
from nltk import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

script_dir = os.path.dirname(__file__)
abs_train_path = os.path.join(script_dir, '../../resources/coset/coset-train.csv')
abs_dev_path = os.path.join(script_dir, '../../resources/coset/coset-dev.csv')
abs_test_tweets_path = os.path.join(script_dir, '../../resources/coset/coset-test-text.csv')
abs_test_truth_path = os.path.join(script_dir, '../../resources/coset/coset-pred-forest.test')
train_ids = []
train_data = []
train_labels = []
# Loading training set
with open(abs_train_path, 'rt', encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
    for row in csv_reader:
        train_ids.append(row[0])
        train_data.append(row[1])
        train_labels.append(row[2])

# Loading validation set
with open(abs_dev_path, 'rt', encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
    for row in csv_reader:
        train_ids.append(row[0])
        train_data.append(row[1])
        train_labels.append(row[2])


def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    #
    # 1. Remove useless things
    p.set_options(p.OPT.URL, p.OPT.NUMBER, p.OPT.SMILEY, p.OPT.EMOJI, p.OPT.RESERVED)
    cleaned_review = p.tokenize(raw_review)
    # if cleaned_review != raw_review:
    #    print(cleaned_review)
    #    print("raw:", raw_review)
    #    print("")
    # 2. Remove non-letters
    # letters_only = re.sub("[^a-zA-Z]", " ", cleaned_review)
    #
    # 3. Convert to lower case, split into individual words
    tokenizer = TweetTokenizer(reduce_len=True)
    words = tokenizer.tokenize(cleaned_review)
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("spanish"))
    #
    # 5. Remove stop words and lower case
    meaningful_words = [w.lower() for w in words if w not in stops]
    #
    # Stemming words
    stemmer = SpanishStemmer()
    stemmed_words = [stemmer.stem(word=word) for word in meaningful_words]
    return " ".join(stemmed_words)


# Initialize an empty list to hold the clean reviews
clean_train_tweets = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list
num_tweets = len(train_data)

for i in range(0, num_tweets):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_tweets.append(review_to_words(train_data[i]))

# vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=8000,binary=True)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', token_pattern=r'\b\w+\b', min_df=1, binary=False)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
train_data_features = vectorizer.fit_transform(clean_train_tweets)

# Numpy arrays are easy to work with, so convert the result to an
# array
train_data_features = train_data_features.toarray()

print("Training the classifier...")

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators=200)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit(train_data_features, train_labels)

# Loading test set
test_ids = []
test_data = []
test_labels = []

with open(abs_test_truth_path) as true_file:
    for line in true_file:
        tweet_id, topic = line.strip().split('\t')
        test_labels.append(int(topic))
        test_ids.append(int(tweet_id))
with open(abs_test_tweets_path, 'rt', encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
    for row in csv_reader:
        test_data.append(str(row[1]))

# Create an empty list and append the clean reviews one by one
num_test_tweets = len(test_data)
clean_test_tweets = []

print("Cleaning and parsing the test set movie reviews...\n")
for i in range(0, num_test_tweets):
    if (i + 1) % 1000 == 0:
        print("Review %d of %d\n" % (i + 1, num_test_tweets))
    clean_tweet = review_to_words(test_data[i])
    clean_test_tweets.append(clean_tweet)

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_tweets)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)
int_result = [int(r) for r in result]
print("int_result", int_result)
print("test_labels", test_labels)
print(f1_score(int_result, test_labels, average='macro'))
