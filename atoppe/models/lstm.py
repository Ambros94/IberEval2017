from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence

from models.model import Model


class LSTMModel(Model):
    def build(self, params):
        self.x_train = sequence.pad_sequences(self.x_train, maxlen=params['maxlen'])
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=params['maxlen'])

        self.model = Sequential()
        self.model.add(Embedding(params['max_features'], params['embedding_dims']))
        self.model.add(
            LSTM(params['lstm_units'], dropout=params['dropout'], recurrent_dropout=params['recurrent_dropout']))
        self.model.add(Dense(self.output_size, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=params['metrics'])
