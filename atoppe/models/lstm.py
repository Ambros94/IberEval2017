from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence

from models.model import Model


class LSTMModel(Model):
    def build(self, params):
        self.x_train = sequence.pad_sequences(self.x_train, maxlen=params['maxlen'])
        self.x_val = sequence.pad_sequences(self.x_val, maxlen=params['maxlen'])
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=params['maxlen'])

        self.model = Sequential()
        self.model.add(Embedding(params['max_features'], 128))
        self.model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(self.output_size, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=params['metrics'])
