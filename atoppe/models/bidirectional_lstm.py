import numpy as np
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.models import Sequential
from keras.preprocessing import sequence

from models.model import Model


class BidirectionalLSTMModel(Model):
    def build(self, params):
        max_features = params['max_features']
        maxlen = params['maxlen']

        self.x_train = sequence.pad_sequences(self.x_train, maxlen=maxlen)
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=maxlen)
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

        self.model = Sequential()
        self.model.add(Embedding(max_features, 128, input_length=maxlen))
        self.model.add(Bidirectional(LSTM(64)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.output_size, activation='sigmoid'))

        self.model.compile('adam', 'binary_crossentropy', metrics=params['metrics'])
