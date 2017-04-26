from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.models import Sequential
from keras.preprocessing import sequence

from models.base import Model
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
