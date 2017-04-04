from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence

from models.base import Model


class CNNModel(Model):

    def build(self, params):
        self.x_train = sequence.pad_sequences(self.x_train, maxlen=params['maxlen'])
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=params['maxlen'])

        self.model = Sequential()

        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        self.model.add(Embedding(params['max_features'],
                                 params['embedding_dims'],
                                 input_length=params['maxlen']))
        self.model.add(Dropout(0.2))

        # we add a Convolution1D, which will learn filters
        # word group filters of size filter_length:
        self.model.add(Conv1D(params['filters'],
                              params['kernel_size'],
                              padding='valid',
                              activation='relu',
                              strides=1))
        # we use max pooling:
        self.model.add(GlobalMaxPooling1D())

        # We add a vanilla hidden layer:
        self.model.add(Dense(params['hidden_dims']))
        self.model.add(Dropout(0.2))
        self.model.add(Activation('relu'))

        # We project onto a single unit output layer, and squash it with a sigmoid:
        self.model.add(Dense(self.output_size))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=params['metrics'])
