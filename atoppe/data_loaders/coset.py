# encoding=utf8
import csv
import os

import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

__author__ = "Ambrosini Luca (@Ambros94)"

script_dir = os.path.dirname(__file__)
abs_train_path = os.path.join(script_dir, '../../resources/coset/coset-train.csv')
abs_dev_path = os.path.join(script_dir, '../../resources/coset/coset-dev.csv')
abs_test_path = os.path.join(script_dir, '../../resources/coset/coset-test-text.csv')

"""
1. Political issues Related to the most abstract electoral confrontation.
2. Policy issues Tweets about sectorial policies.
9. Campaign issues Related with the evolution of the campaign.
10. Personal issues The topic is the personal life and activities of the candidates.
11. Other issues.
"""
label_encoder = None


def load_data():
    """
    Loads coset training and dev data sets
    :return: (ids_train, x_train, y_train),(ids_test, x_test, y_test)
    """
    ids = []
    data = []
    labels = []
    training_samples, validation_samples = 0, 0
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

    # Prepare labels
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_y = encoder.transform(labels)
    ready_y = np_utils.to_categorical(encoded_y)

    print('Average sequence length: {} chars'.format(np.mean(list(map(len, data)), dtype=int)))
    print('Max sequence length: {} chars'.format(np.max(list(map(len, data)))))
    return ids, data, ready_y


def load_test():
    ids = []
    data = []
    # Loading test set
    with open(abs_test_path, 'rt', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        for row in csv_reader:
            ids.append(row[0])
            data.append(row[1])
    return ids, data


def decode_label(label):
    decoded = \
        {0: 1,
         3: 2,
         4: 9,
         1: 10,
         2: 11}.get(label.argmax(), "Error")
    return decoded


def decode_labels(labels):
    decoded_labels = []
    for label in labels:
        decoded_labels.append(decode_label(label))
    return decoded_labels


def persist_solution(ids, labels, filename="coset.txt"):
    decoded_labels = decode_labels(labels)

    with open(filename, 'w') as out_file:
        for id, label in zip(ids, decoded_labels):
            out_file.write("{id}\t{label}\n".format(id=id, label=label))
