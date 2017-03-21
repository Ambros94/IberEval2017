import csv
import os

script_dir = os.path.dirname(__file__)
abs_train_path = os.path.join(script_dir, '../resources/coset-train.csv')
abs_dev_path = os.path.join(script_dir, '../resources/coset-dev.csv')


def load_data():
    """
    Loads data form file, the train set contains also the dev
    :return: (X_train, y_train), (X_test, y_test)
    """
    X_train, y_train, X_test, y_test = [], [], [], []
    with open(abs_train_path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            X_train.append(row[1])
            y_train.append(row[2])

    with open(abs_dev_path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            X_train.append(row[1])
            y_train.append(row[2])

    return (X_train[1:], y_train[1:]), (X_test, y_test)

# (X_train, y_train), (X_test, y_test) = load_data()
# print(X_train[0], y_train[0])
# print len(X_train)
