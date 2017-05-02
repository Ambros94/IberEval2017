# encoding=utf8
import csv
import os

import keras.backend as K
import numpy as np
import preprocessor as p
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

__author__ = "Ambrosini Luca (@Ambros94)"

script_dir = os.path.dirname(__file__)
abs_train_path = os.path.join(script_dir, '../../resources/coset/coset-train.csv')
abs_dev_path = os.path.join(script_dir, '../../resources/coset/coset-dev.csv')
abs_test_tweets_path = os.path.join(script_dir, '../../resources/coset/coset-test-text.csv')
abs_test_truth_path = os.path.join(script_dir, '../../resources/coset/coset-pred-forest.test')

"""
1. Political issues Related to the most abstract electoral confrontation.
2. Policy issues Tweets about sectorial policies.
9. Campaign issues Related with the evolution of the campaign.
10. Personal issues The topic is the personal life and activities of the candidates.
11. Other issues.
"""
label_encoder = None


def fbeta_score(y_true, y_pred, beta=1):
    """
    Computes the F score, the weighted harmonic mean of precision and recall.
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
    """
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


def load_data(max_words=10000, char_level=False, pre_process=False):
    """
    Loads data form file, the train set contains also the dev
    :param pre_process: Use pre-processor to tokenize tweets
    :param char_level: If True char_level tokenizer is used
    :param max_words: Max number of words that are considered (Most used words in corpus)
    :return: (ids_train, x_train, y_train),(ids_test, x_test, y_test)
    """
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

    if pre_process:
        p.set_options(p.OPT.URL, p.OPT.RESERVED, p.OPT.MENTION, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)
        data = [p.tokenize(d) for d in data]
    # Prepare data
    tokenizer = Tokenizer(num_words=max_words, char_level=char_level)
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
    # Test
    ids_test = ids[training_samples + validation_samples:]
    x_test = data[training_samples + validation_samples:]
    y_test = ready_y[training_samples + validation_samples:]

    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

    print('Max train sequence length: {}'.format(np.max(list(map(len, x_train)))))
    print('Max test sequence length: {}'.format(np.max(list(map(len, x_test)))))

    return (ids_train, x_train, y_train), (ids_test, x_test, y_test)


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


def persist_solution(ids, labels):
    decoded_labels = decode_labels(labels)

    with open('results.txt', 'w') as out_file:
        for id, label in zip(ids, decoded_labels):
            out_file.write("{id}\t{label}\n".format(id=id, label=label))
