# encoding=utf8
import csv
import os

from keras.preprocessing.text import Tokenizer

script_dir = os.path.dirname(__file__)
abs_train_path = os.path.join(script_dir, '../resources/coset-train.csv')
abs_dev_path = os.path.join(script_dir, '../resources/coset-dev.csv')


class DataLoader:
    def __init__(self):
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

    def load_data(self):
        """
        Loads data form file, the train set contains also the dev
        :return: (X_train, y_train), (X_test, y_test)
        """

        # Train set
        with open(abs_train_path, 'rt', encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in csv_reader:
                self.X_train.append(row[1])
                self.y_train.append(row[2])
        # Dev set
        with open(abs_dev_path, 'rt', encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in csv_reader:
                self.X_train.append(row[1])
                self.y_train.append(row[2])

        self.tokenize()

        return (self.X_train, self.y_train), (self.X_test, self.y_test)

    def tokenize(self):
        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(self.X_train)
        self.X_train = tokenizer.texts_to_sequences(self.X_train)


(X_train, y_train), (X_test, y_test) = DataLoader().load_data()
print(X_train[0], y_train[0])
