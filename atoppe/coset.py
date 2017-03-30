import abc

from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence

from data_loaders import stance


class Model:
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_function):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data_function()
        if len(self.y_train) == 0:
            raise Exception("You should provide at least one train label")
        self.output_size = len(self.y_train[0])
        self.model = None

    def run(self, batch_size, epochs, **params):
        self.build(params)
        self.train(batch_size=batch_size, epochs=epochs)
        return self.evaluate(batch_size=batch_size)

    @abc.abstractmethod
    def build(self, params):
        """Build a model providing all necessary parameters"""
        return

    def train(self, batch_size, epochs):
        self.model.fit(self.x_train, self.y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(self.x_test, self.y_test))

    def evaluate(self, batch_size):
        return self.model.evaluate(self.x_test, self.y_test,
                                   batch_size=batch_size)


class CNNModel(Model):
    def build(self, params):
        print('Pad sequences (samples x time)')
        self.x_train = sequence.pad_sequences(self.x_train, maxlen=params['maxlen'])
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=params['maxlen'])
        print('x_train shape:', self.x_train.shape)
        print('x_test shape:', self.x_test.shape)
        print('y_train shape:', self.y_train.shape)
        print('y_test shape:', self.y_test.shape)

        print('Build model...')
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
                           metrics=['categorical_accuracy'])


cnn = CNNModel(stance.load_data)
score, acc = cnn.run(max_features=15000, maxlen=50, batch_size=32, embedding_dims=50, filters=250, kernel_size=3,
                     hidden_dims=250, epochs=5)
print('Test score:', score)
print('Test accuracy:', acc)
