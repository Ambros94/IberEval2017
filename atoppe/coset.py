# encoding=utf8
import csv
import os

import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

__author__ = "Ambrosini Luca (@Ambros94)"

script_dir = os.path.dirname(__file__)
abs_train_path = os.path.join(script_dir, '../resources/coset-train.csv')
abs_dev_path = os.path.join(script_dir, '../resources/coset-dev.csv')

"""
1. Political issues Related to the most abstract electoral confrontation.
2. Policy issues Tweets about sectorial policies.
9. Campaign issues Related with the evolution of the campaign.
10. Personal issues The topic is the personal life and activities of the candidates.
11. Other issues.
"""


def load_data(max_words=15000, n_validation_samples=250, verbose=False, debug=False):
    """
    Loads data form file, the train set contains also the dev
    :return: (X_train, y_train), (X_test, y_test)
    """
    data = []
    labels = []
    # Loading
    with open(abs_train_path, 'rt', encoding="utf-8") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in csv_reader:
            data.append(row[1])
            labels.append(row[2])
    train_size = len(data)
    if verbose:
        print("Loaded {n_examples} examples from train set".format(n_examples=train_size))
    with open(abs_dev_path, 'rt', encoding="utf-8") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in csv_reader:
            data.append(row[1])
            labels.append(row[2])
    if verbose:
        print("Loaded {n_examples} examples from dev set".format(n_examples=len(data) - train_size))
    # Keep only some very small data so it's easier to manually analise
    if debug:
        data = data[:3]
        labels = labels[:3]
    if verbose:
        print("Loaded data-set:")
        for i, d in enumerate(data):
            print("{label}\t:\t{data}".format(data=d, label=labels[i]))
    # Prepare data
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data)
    data = tokenizer.texts_to_sequences(data)
    print('Found {word_index} unique tokens: {words}'.format(word_index=len(tokenizer.word_index),
                                                             words=tokenizer.word_index))
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
    if verbose:
        print("Pre-processed data-set:")
        for i, d in enumerate(x_train):
            print("{label}\t:\t{data}".format(data=d, label=y_train[i]))
    print(x_train[0], x_val[0])
    print(x_train[-1], x_val[-1])
    return (x_train, y_train), (x_val, y_val)


def coset_f1(y_true, y_predicted):
    # TODO Implement the correct metrix used in the test instead of normal categorical_accuracy
    return K.mean(y_predicted)
