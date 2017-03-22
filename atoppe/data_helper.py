# encoding=utf8
import csv
import os
import sys

from spacy.es import Spanish

# Fix encoding problems
reload(sys)
sys.setdefaultencoding('utf8')

script_dir = os.path.dirname(__file__)
abs_train_path = os.path.join(script_dir, '../resources/coset-train.csv')
abs_dev_path = os.path.join(script_dir, '../resources/coset-dev.csv')

nlp = Spanish(path=None)


def load_data():
    """
    Loads data form file, the train set contains also the dev
    :return: (X_train, y_train), (X_test, y_test)
    """
    X_train, y_train, X_test, y_test = [], [], [], []
    # Train set
    with open(abs_train_path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            X_train.append(row[1])
            y_train.append(row[2])
    # Dev set
    with open(abs_dev_path, 'rb') as csvfile
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            X_train.append(row[1])
            y_train.append(row[2])

    # TODO Go on with data pre-processing, at least join hash tags in a single word
    # p.s Hashtags are not in the dictionary anymore, we should come up with something
    #  #perro is not in the dict, perro is

    return (X_train[1:], y_train[1:]), (X_test, y_test)


def data_statistics(sentences):
    for s in sentences:
        tokens = nlp(unicode(s))
        print("************")
        print("Full sentence", unicode(s))
        for t in tokens:
            print(t)


(X_train, y_train), (X_test, y_test) = load_data()
data_statistics(X_train)
