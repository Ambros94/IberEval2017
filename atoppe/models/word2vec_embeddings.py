'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classication of newsgroup messages into 20 different categories).

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function

import csv
import os

import numpy as np
import preprocessor as p
from keras.layers import Dense, GlobalAveragePooling1D
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from data_loaders import coset

MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
script_dir = os.path.dirname(__file__)
abs_train_path = '/Users/lambrosini/PycharmProjects/IberEval2017/resources/coset/coset-train.csv'
abs_dev_path = '/Users/lambrosini/PycharmProjects/IberEval2017/resources/coset/coset-dev.csv'
abs_test_tweets_path = '/Users/lambrosini/PycharmProjects/IberEval2017/resources/coset/coset-test-text.csv'
abs_test_truth_path = '/Users/lambrosini/PycharmProjects/IberEval2017/resources/coset/coset-pred-forest.test'
# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open("/Users/lambrosini/PycharmProjects/IberEval2017/es/es.vec")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

ids = []
data = []
labels = []
training_samples, validation_samples, test_samples_1, test_samples_2 = 0, 0, 0, 0
# Loading training set
with open(abs_train_path, 'rt', encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
    for row in csv_reader:
        ids.append(row[0])
        data.append(row[1])
        labels.append(row[2])
        training_samples += 1
# Loading validation set
with open(abs_dev_path, 'rt', encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
    for row in csv_reader:
        ids.append(row[0])
        data.append(row[1])
        labels.append(row[2])
        validation_samples += 1

# Loading test set
with open(abs_test_truth_path) as true_file:
    for line in true_file:
        tweet_id, topic = line.strip().split('\t')
        labels.append(int(topic))
        ids.append(int(tweet_id))
        test_samples_1 += 1
with open(abs_test_tweets_path, 'rt', encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
    for row in csv_reader:
        data.append(row[1])
        test_samples_2 += 1

p.set_options(p.OPT.URL, p.OPT.RESERVED, p.OPT.MENTION, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)
data = [p.tokenize(d) for d in data]
# Prepare data
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(data)
data = tokenizer.texts_to_sequences(data)
print('Found {word_index} unique tokens'.format(word_index=len(tokenizer.word_index)))

# Prepare labels
encoder = LabelEncoder()
encoder.fit(labels)
encoded_y = encoder.transform(labels)
ready_y = np_utils.to_categorical(encoded_y)

# Train
ids_train = ids[0:training_samples + validation_samples]
x_train = data[0:training_samples + validation_samples]
y_train = ready_y[0:training_samples + validation_samples]
# Dev
ids_test = ids[training_samples + validation_samples:]
x_test = data[training_samples + validation_samples:]
y_test = ready_y[training_samples + validation_samples:]

word_index = tokenizer.word_index

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
found, oob = 0, 0
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        found += 1
    else:
        oob += 1
print("Found", found)
print("OOB", oob)
# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

print('Training model.')

x_train = sequence.pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)

# train a 1D convnet with global maxpooling
model = Sequential()
model.add(Embedding(num_words,
                    EMBEDDING_DIM,
                    weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=True))
model.add(GlobalAveragePooling1D())

model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[coset.fbeta_score])

model.fit(x_train, y_train,
          batch_size=32,
          epochs=4,
          validation_data=(x_test, y_test))
predictions = model.predict(x_test, batch_size=32)
print(f1_score(coset.decode_labels(y_test), coset.decode_labels(predictions), average='macro'))