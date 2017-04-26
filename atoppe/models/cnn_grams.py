from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence

from models.base import Model
from nlp_utils.n_grams import augment_with_n_grams


class CNNGramsModel(Model):
    def build(self, params):
        max_len = params['maxlen']
        max_features = params['max_features']
        embedding_dims = params['embedding_dims']
        filters = params['filters']
        kernel_size = params['kernel_size']
        hidden_dims = params['hidden_dims']
        metrics = params['metrics']
        ngram_range = params['ngram_range']

        self.x_train, self.x_test, max_features = augment_with_n_grams(x_train=self.x_train, x_test=self.x_test,
                                                                       max_features=max_features,
                                                                       ngram_range=ngram_range)

        self.x_train = sequence.pad_sequences(self.x_train, maxlen=max_len)
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=max_len)

        self.model = Sequential()

        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        self.model.add(Embedding(max_features,
                                 embedding_dims,
                                 input_length=max_len))
        self.model.add(Dropout(0.2))

        # we add a Convolution1D, which will learn filters
        # word group filters of size filter_length:
        self.model.add(Conv1D(filters=filters,
                              kernel_size=kernel_size,
                              padding='valid',
                              activation='relu',
                              strides=1))
        # we use max pooling:
        self.model.add(GlobalMaxPooling1D())

        # We add a vanilla hidden layer:
        self.model.add(Dense(hidden_dims))
        self.model.add(Dropout(0.2))
        self.model.add(Activation('relu'))

        # We project onto a single unit output layer, and squash it with a sigmoid:
        self.model.add(Dense(self.output_size))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=metrics)
