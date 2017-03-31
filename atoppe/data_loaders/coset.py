# encoding=utf8
import csv
import os

import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

__author__ = "Ambrosini Luca (@Ambros94)"

script_dir = os.path.dirname(__file__)
abs_train_path = os.path.join(script_dir, '../../resources/coset/coset-train.csv')
abs_dev_path = os.path.join(script_dir, '../../resources/coset/coset-dev.csv')

"""
1. Political issues Related to the most abstract electoral confrontation.
2. Policy issues Tweets about sectorial policies.
9. Campaign issues Related with the evolution of the campaign.
10. Personal issues The topic is the personal life and activities of the candidates.
11. Other issues.
"""


def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    '''Computes the F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification where input samples can be
    tagged with a set of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Weight precision and recall together as a single scalar.
    beta2 = beta ** 2
    f_score = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)
    return f_score


def load_data(max_words=15000, n_validation_samples=250):
    """
    Loads data form file, the train set contains also the dev
    :param max_words: Max number of words that are considered (Most used words in corpus)
    :param n_validation_samples: How many examples have to go from the data-set into the test set
    :return: (x_train, y_train), (x_test, y_test)
    """
    data = []
    labels = []
    # Loading
    with open(abs_train_path, 'rt', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        for row in csv_reader:
            data.append(row[1])
            labels.append(row[2])
    with open(abs_dev_path, 'rt', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        for row in csv_reader:
            data.append(row[1])
            labels.append(row[2])
    # Prepare data
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data)
    data = tokenizer.texts_to_sequences(data)
    # print('Found {word_index} unique tokens'.format(word_index=len(tokenizer.word_index)))

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


def coset_f1(y_true, y_predicted):
    # TODO Implement the correct metrix used in the test instead of normal categorical_accuracy
    return K.mean(y_predicted)
