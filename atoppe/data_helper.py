# encoding=utf8
import csv
import os

from keras.preprocessing.text import Tokenizer

script_dir = os.path.dirname(__file__)
abs_train_path = os.path.join(script_dir, '../resources/coset-train.csv')
abs_dev_path = os.path.join(script_dir, '../resources/coset-dev.csv')


class DataLoader:
    def __init__(self):
        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []

    def load_data(self, nb_validation_samples=500):
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
        with open(abs_dev_path, 'rt', encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in csv_reader:
                data.append(row[1])
                labels.append(row[2])

        # Prepare
        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(data)
        data = tokenizer.texts_to_sequences(data)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        # Split in train and test
        self.x_train = data[:-nb_validation_samples]
        self.y_train = labels[:-nb_validation_samples]
        self.x_val = data[-nb_validation_samples:]
        self.y_val = labels[-nb_validation_samples:]

        return (self.x_train, self.y_train), (self.x_val, self.y_val)
