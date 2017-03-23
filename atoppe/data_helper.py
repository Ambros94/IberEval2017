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


class DataLoader:
    def __init__(self):
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

    def load_data(self, tokenize=True, join_hash_tags=False):
        """
        Loads data form file, the train set contains also the dev
        :return: (X_train, y_train), (X_test, y_test)
        """

        # Train set
        with open(abs_train_path, 'rb') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in csv_reader:
                self.X_train.append(row[1])
                self.y_train.append(row[2])
        # Dev set
        with open(abs_dev_path, 'rb') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in csv_reader:
                self.X_train.append(row[1])
                self.y_train.append(row[2])
        if tokenize:
            self.X_train = DataLoader.tokenize(self.X_train)
            self.X_test = DataLoader.tokenize(self.X_test)
        if join_hash_tags:
            self.join_hash_tags()

        return (self.X_train, self.y_train), (self.X_test, self.y_test)

    @classmethod
    def tokenize(cls, sentences_array):
        """
        Tokenize the sentence array given as input
        :param sentences_array: Iterable of sentences that will be tokenize
        :return: Array of arrays containing tokenized sentences
        """
        tokenize_array = []
        for s in sentences_array:
            tokenize_array.append([t for t in nlp(unicode(s))])
        return tokenize_array

    def join_hash_tags(self):
        pass


(X_train, y_train), (X_test, y_test) = DataLoader().load_data(tokenize=True)
print(X_train[0], y_train[0])
