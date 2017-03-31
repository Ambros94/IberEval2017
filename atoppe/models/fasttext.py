import numpy as np
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.models import Sequential
from keras.preprocessing import sequence

from models.base import Model


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.

    {(4, 9), (4, 1), (1, 4), (9, 4)}

    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.

    Example: adding bi-gram
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    Example: adding tri-gram
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


class FastTextModel(Model):
    def build(self, params):
        ngram_range = params['ngram_range']
        max_features = params['max_features']
        maxlen = params['maxlen']
        embedding_dims = params['embedding_dims']

        if ngram_range > 1:
            print('Adding {}-gram features'.format(ngram_range))
            # Create set of unique n-gram from the training set.
            ngram_set = set()
            for input_list in self.x_train:
                for i in range(2, ngram_range + 1):
                    set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                    ngram_set.update(set_of_ngram)

            # Dictionary mapping n-gram token to a unique integer.
            # Integer values are greater than max_features in order
            # to avoid collision with existing features.
            start_index = max_features + 1
            token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
            indice_token = {token_indice[k]: k for k in token_indice}

            # max_features is the highest integer that could be found in the dataset.
            max_features = np.max(list(indice_token.keys())) + 1

            # Augmenting X_train and X_test with n-grams features
            self.x_train = add_ngram(self.x_train, token_indice, ngram_range)
            self.x_test = add_ngram(self.x_test, token_indice, ngram_range)
            print('Average train sequence length: {}'.format(np.mean(list(map(len, self.x_train)), dtype=int)))
            print('Average test sequence length: {}'.format(np.mean(list(map(len, self.x_test)), dtype=int)))
            print('Min test sequence length: {}'.format(np.min(list(map(len, self.x_test)))))

        self.x_train = sequence.pad_sequences(self.x_train, maxlen=maxlen)
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=maxlen)
        self.model = Sequential()

        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        self.model.add(Embedding(max_features,
                                 embedding_dims,
                                 input_length=maxlen))

        # we add a GlobalAveragePooling1D, which will average the embeddings
        # of all words in the document
        self.model.add(GlobalAveragePooling1D())

        # We project onto a single unit output layer, and squash it with a sigmoid:
        self.model.add(Dense(self.output_size, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=params['metrics'])
