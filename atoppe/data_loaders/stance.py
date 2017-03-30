# encoding=utf8
import csv
import os

import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

__author__ = "Ambrosini Luca (@Ambros94)"

script_dir = os.path.dirname(__file__)
abs_truth_path = os.path.join(script_dir, '../../resources/stance/alc-truth.csv')
abs_tweets_path = os.path.join(script_dir, '../../resources/stance/alc-tweets.csv')

"""
1. Political issues Related to the most abstract electoral confrontation.
2. Policy issues Tweets about sectorial policies.
9. Campaign issues Related with the evolution of the campaign.
10. Personal issues The topic is the personal life and activities of the candidates.
11. Other issues.
"""


def load_data(max_words=15000, n_validation_samples=5):
    """
    Loads data form file, the train set contains also the dev
    :param max_words: Max number of words that are considered (Most used words in corpus)
    :param n_validation_samples: How many examples have to go from the data-set into the test set
    :return: (x_train, y_train), (x_test, y_test)
    """
    data = []
    labels = []
    # Loading data
    with open(abs_truth_path, 'rt', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            labels.append(row[1] + row[2])
    with open(abs_tweets_path, 'rt', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            data.append(row[1])

    # Prepare data
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data)
    data = tokenizer.texts_to_sequences(data)
    print('Found {word_index} unique tokens'.format(word_index=len(tokenizer.word_index)))

    # Prepare labels
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_y = encoder.transform(labels)
    ready_y = np_utils.to_categorical(encoded_y)

    # Split in train and test
    x_train = data[:-n_validation_samples]
    y_train = ready_y[:-n_validation_samples]
    x_val = data[-n_validation_samples:]
    y_val = ready_y[-n_validation_samples:]
    return (x_train, y_train), (x_val, y_val)


def stance_metric(y_true, y_predicted):
    # TODO Implement the correct metrix used in the test instead of normal categorical_accuracy
    return K.mean(y_predicted)
