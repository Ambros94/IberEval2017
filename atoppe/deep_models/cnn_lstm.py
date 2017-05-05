from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from deep_models.toppemodel import ToppeModel
from nlp_utils.tweets_preprocessor import clean_tweets


class CnnLstmModel(ToppeModel):
    def build(self, params):
        # Parameters
        maxlen = params['maxlen']
        embedding_size = params['embedding_size']
        kernel_size = params['kernel_size']
        filters = params['filters']
        pool_size = params['pool_size']
        strides = params['strides']
        lstm_output_size = params['lstm_output_size']
        dropout = params['dropout']

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
        self.keras_model.add(Embedding(num_words, embedding_size, input_length=maxlen))
        self.keras_model.add(Dropout(dropout))
        self.keras_model.add(Conv1D(filters,
                                    kernel_size,
                                    padding='valid',
                                    activation='relu',
                                    strides=strides))
        self.keras_model.add(MaxPooling1D(pool_size=pool_size))
        self.keras_model.add(LSTM(lstm_output_size))
        self.keras_model.add(Dense(self.output_size))
        self.keras_model.add(Activation('softmax'))

        self.keras_model.compile(loss='categorical_crossentropy',
                                 optimizer='adam',
                                 metrics=params['metrics'])
