from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.models import Sequential
from keras.preprocessing import sequence

from models.model import Model
from nlp_utils.n_grams import augment_with_n_grams


class FastTextModel(Model):
    def build(self, params):
        ngram_range = params['ngram_range']
        max_features = params['max_features']
        maxlen = params['maxlen']
        embedding_dims = params['embedding_dims']

        self.x_train, self.x_test, max_features = augment_with_n_grams(x_train=self.x_train, x_test=self.x_test,
                                                                       max_features=max_features,
                                                                       ngram_range=ngram_range)

        self.x_train = sequence.pad_sequences(self.x_train, maxlen=params['maxlen'])
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=params['maxlen'])
        self.model = Sequential()

        self.model.add(Embedding(max_features,
                                 embedding_dims,
                                 input_length=maxlen))

        self.model.add(GlobalAveragePooling1D())

        self.model.add(Dense(self.output_size, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=params['metrics'])
