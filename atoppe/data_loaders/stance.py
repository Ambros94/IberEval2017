# encoding=utf8
import csv
import os
from random import shuffle

import numpy
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

__author__ = "Ambrosini Luca (@Ambros94)"

script_dir = os.path.dirname(__file__)
abs_truth_ca_path = os.path.join(script_dir, '../../resources/stance/training_truth_ca.txt')
abs_tweets_ca_path = os.path.join(script_dir, '../../resources/stance/training_tweets_ca.txt')

abs_truth_es_path = os.path.join(script_dir, '../../resources/stance/training_truth_es.txt')
abs_tweets_es_path = os.path.join(script_dir, '../../resources/stance/training_tweets_es.txt')

"""
1. Political issues Related to the most abstract electoral confrontation.
2. Policy issues Tweets about sectorial policies.
9. Campaign issues Related with the evolution of the campaign.
10. Personal issues The topic is the personal life and activities of the candidates.
11. Other issues.
"""


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    # Given list1 and list2
    list1_shuf = []
    list2_shuf = []
    index_shuf = list(range(len(a)))
    shuffle(index_shuf)
    for i in index_shuf:
        list1_shuf.append(a[i])
        list2_shuf.append(b[i])
    return list1_shuf, list2_shuf


def load_data(n_validation_samples=250):
    """
    Loads data form file, the train set contains also the dev
    :param n_validation_samples: How many examples have to go from the data-set into the test set
    :return: (x_train, y_train), (x_test, y_test)
    """
    ids = []
    data = []
    labels = []
    # Loading ca
    with open(abs_truth_ca_path, 'rt', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            ids.append(row[0])
            labels.append(row[1] + row[2])
    with open(abs_tweets_ca_path, 'rt', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            if len(row) > 2:
                data.append(';'.join(row[1:]))
            else:
                data.append(row[1])

    with open(abs_truth_es_path, 'rt', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            ids.append(row[0])
            labels.append(row[1] + row[2])
    with open(abs_tweets_es_path, 'rt', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for i, row in enumerate(csv_reader):
            if len(row) >= 2:
                data.append(';'.join(row[1:]))
            else:
                data.append(row[1])

    data, labels = unison_shuffled_copies(data, labels)

    # Prepare labels
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_y = encoder.transform(labels)
    ready_y = np_utils.to_categorical(encoded_y)

    # Train
    ids_train = ids[:-n_validation_samples]
    x_train = data[:-n_validation_samples]
    y_train = ready_y[:-n_validation_samples]
    # Validation
    ids_val = ids[-n_validation_samples:]
    x_val = data[-n_validation_samples:]
    y_val = ready_y[-n_validation_samples:]
    return (ids_train, x_train, y_train), (ids_val, x_val, y_val)


def load_stance_es(n_validation_samples=250):
    ids = []
    data = []
    labels = []
    # Loading ca
    with open(abs_truth_es_path, 'rt', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            ids.append(row[0])
            labels.append(row[1])
    with open(abs_tweets_es_path, 'rt', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            if len(row) > 2:
                data.append(';'.join(row[1:]))
            else:
                data.append(row[1])
    assert len(ids) == 4319
    assert len(data) == 4319
    assert len(labels) == 4319
    # Prepare labels
    ready_y = [encode_stance(l) for l in labels]
    # Train
    ids_train = ids[:-n_validation_samples]
    x_train = data[:-n_validation_samples]
    y_train = ready_y[:-n_validation_samples]
    # Validation
    ids_val = ids[-n_validation_samples:]
    x_val = data[-n_validation_samples:]
    y_val = ready_y[-n_validation_samples:]
    return (ids_train, x_train, y_train), (ids_val, x_val, y_val)


def load_stance_ca(n_validation_samples=250):
    ids = []
    data = []
    labels = []
    # Loading ca
    with open(abs_truth_ca_path, 'rt', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            ids.append(row[0])
            labels.append(row[1])
    with open(abs_tweets_ca_path, 'rt', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            if len(row) > 2:
                data.append(';'.join(row[1:]))
            else:
                data.append(row[1])

    # Prepare labels
    ready_y = [encode_stance(l) for l in labels]
    # Train
    ids_train = ids[:-n_validation_samples]
    x_train = data[:-n_validation_samples]
    y_train = ready_y[:-n_validation_samples]
    # Validation
    ids_val = ids[-n_validation_samples:]
    x_val = data[-n_validation_samples:]
    y_val = ready_y[-n_validation_samples:]
    return (ids_train, x_train, y_train), (ids_val, x_val, y_val)


def load_gender_es(n_validation_samples=250):
    ids = []
    data = []
    labels = []
    # Loading ca
    with open(abs_truth_es_path, 'rt', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            ids.append(row[0])
            labels.append(row[2])
    with open(abs_tweets_es_path, 'rt', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            if len(row) > 2:
                data.append(';'.join(row[1:]))
            else:
                data.append(row[1])
    # Prepare labels
    ready_y = [encode_gender(l) for l in labels]

    # Train
    ids_train = ids[:-n_validation_samples]
    x_train = data[:-n_validation_samples]
    y_train = ready_y[:-n_validation_samples]
    # Validation
    ids_val = ids[-n_validation_samples:]
    x_val = data[-n_validation_samples:]
    y_val = ready_y[-n_validation_samples:]
    return (ids_train, x_train, y_train), (ids_val, x_val, y_val)


def load_gender_ca(n_validation_samples=400):
    ids = []
    data = []
    labels = []
    # Loading ca
    with open(abs_truth_ca_path, 'rt', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            ids.append(row[0])
            labels.append(row[2])
    with open(abs_tweets_ca_path, 'rt', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            if len(row) > 2:
                data.append(';'.join(row[1:]))
            else:
                data.append(row[1])

    # Prepare labels
    ready_y = [encode_gender(l) for l in labels]
    # Train
    ids_train = ids[:-n_validation_samples]
    x_train = data[:-n_validation_samples]
    y_train = ready_y[:-n_validation_samples]
    # Validation
    ids_val = ids[-n_validation_samples:]
    x_val = data[-n_validation_samples:]
    y_val = ready_y[-n_validation_samples:]
    return (ids_train, x_train, y_train), (ids_val, x_val, y_val)


def encode_stance(label):
    return {'AGAINST': [1., 0., 0.],
            'NEUTRAL': [0., 1., 0.],
            'FAVOR': [0., 0., 1.]}.get(label, 'Error')


def decode_stance(label):
    return {0: 'AGAINST',
            1: 'NEUTRAL',
            2: 'FAVOR'}.get(numpy.array(label).argmax(), "Error")


def encode_gender(label):
    return {'FEMALE': [1., 0.],
            'MALE': [0., 1.]}.get(label, 'Error')


def decode_gender(label):
    y = {0: 'FEMALE',
         1: 'MALE', }.get(numpy.array(label).argmax(), "Error")
    return y


def persist_gender():
    pass  # TODO Implement


def persist_stance():
    pass  # TODO Implement
