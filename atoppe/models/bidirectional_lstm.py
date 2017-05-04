import numpy as np
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from keras.models import Sequential
from keras.preprocessing import sequence

from models.mymodel import MyModel


class BidirectionalLSTMModel(MyModel):
    def build(self, params):
        max_features = params['max_features']
        max_len = params['max_len']
        embedding_dims = params['embedding_dims']
        recurrent_units = params['recurrent_units']
        dropout = params['dropout']

        self.x_train = sequence.pad_sequences(self.x_train, maxlen=params['max_len'])
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=params['max_len'])
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

        self.model = Sequential()
        self.model.add(Embedding(max_features, embedding_dims, input_length=max_len))
        self.model.add(Bidirectional(LSTM(recurrent_units)))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(self.output_size, activation='softmax'))

        self.model.compile('adam', 'categorical_crossentropy', metrics=params['metrics'])
