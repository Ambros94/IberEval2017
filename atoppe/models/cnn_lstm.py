from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence

from models.model import Model


class CnnLstmModel(Model):
    def build(self, params):
        max_features = params['max_features']
        max_len = params['maxlen']
        embedding_size = params['embedding_size']

        # Convolution
        kernel_size = params['kernel_size']
        filters = params['filters']
        pool_size = params['pool_size']

        # LSTM
        lstm_output_size = params['lstm_output_size']

        self.x_train = sequence.pad_sequences(self.x_train, maxlen=params['maxlen'])
        self.x_val = sequence.pad_sequences(self.x_val, maxlen=params['maxlen'])
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=params['maxlen'])

        self.model = Sequential()
        self.model.add(Embedding(max_features, embedding_size, input_length=max_len))
        self.model.add(Dropout(0.25))
        self.model.add(Conv1D(filters,
                              kernel_size,
                              padding='valid',
                              activation='relu',
                              strides=1))
        self.model.add(MaxPooling1D(pool_size=pool_size))
        self.model.add(LSTM(lstm_output_size))
        self.model.add(Dense(self.output_size))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=params['metrics'])
