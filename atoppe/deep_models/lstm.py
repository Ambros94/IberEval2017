from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from deep_models.toppemodel import ToppeModel
from nlp_utils.tweets_preprocessor import clean_tweets


class LSTMModel(ToppeModel):
    def build(self, params):
        # Extract params
        maxlen = params['maxlen']
        # Cleaning data
        self.x_train = clean_tweets(self.x_train)
        self.x_test = clean_tweets(self.x_test)
        # Prepare data
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.x_train)
        num_words = len(tokenizer.word_index) + 1
        x_train = tokenizer.texts_to_sequences(self.x_train)
        x_test = tokenizer.texts_to_sequences(self.x_test)
        print('Found {word_index} words'.format(word_index=num_words))
        self.x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        self.x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

        self.keras_model = Sequential()
        self.keras_model.add(Embedding(num_words, params['embedding_dims']))
        self.keras_model.add(
            LSTM(params['lstm_units'], dropout=params['dropout'], recurrent_dropout=params['recurrent_dropout']))
        self.keras_model.add(Dense(self.output_size, activation='softmax'))

        self.keras_model.compile(loss='categorical_crossentropy',
                                 optimizer='adam',
                                 metrics=params['metrics'])
