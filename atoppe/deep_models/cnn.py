from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from deep_models.toppemodel import ToppeModel
from nlp_utils.tweets_preprocessor import clean_tweets


class CNNModel(ToppeModel):
    def build(self, params):
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
        self.x_train = sequence.pad_sequences(x_train, maxlen=params['maxlen'])
        self.x_test = sequence.pad_sequences(x_test, maxlen=params['maxlen'])
        # Build model
        self.keras_model = Sequential()
        self.keras_model.add(Embedding(num_words,
                                       params['embedding_dims'],
                                       input_length=params['maxlen']))
        self.keras_model.add(Dropout(params['dropout']))
        self.keras_model.add(Conv1D(params['filters'],
                                    params['kernel_size'],
                                    padding='same',
                                    activation='relu',
                                    strides=params['strides']))
        self.keras_model.add(GlobalMaxPooling1D())
        self.keras_model.add(Dense(params['hidden_dims']))
        self.keras_model.add(Dropout(params['dropout_final']))
        self.keras_model.add(Activation('relu'))
        self.keras_model.add(Dense(self.output_size))
        self.keras_model.add(Activation('softmax'))
        # Compile
        self.keras_model.compile(loss='categorical_crossentropy',
                                 optimizer='adam', metrics=params['metrics'])
