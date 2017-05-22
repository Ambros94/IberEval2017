from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from deep_models.toppemodel import ToppeModel
from nlp_utils import word_vectors
from nlp_utils.tweets_preprocessor import clean_tweets


class BidirectionalLSTMModel(ToppeModel):
    def build(self, params):
        # Extract params
        max_len = params['max_len']
        language = params['language']
        recurrent_units = params['recurrent_units']
        dropout = params['dropout']
        # Cleaning data
        clean_function = params['clean_tweets']
        self.x_train = clean_tweets(cleaning_function=clean_function,tweets=self.x_train)
        self.x_test = clean_tweets(cleaning_function=clean_function,tweets=self.x_test)
        self.x_persist = clean_tweets(cleaning_function=clean_function, tweets=self.x_persist)
        # Prepare data
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.x_train)
        num_words = len(tokenizer.word_index) + 1
        x_train = tokenizer.texts_to_sequences(self.x_train)
        x_test = tokenizer.texts_to_sequences(self.x_test)
        x_persist = tokenizer.texts_to_sequences(self.x_persist)
        print('Found {word_index} words'.format(word_index=num_words))
        self.x_train = sequence.pad_sequences(x_train, maxlen=max_len)
        self.x_test = sequence.pad_sequences(x_test, maxlen=max_len)
        self.x_persist = sequence.pad_sequences(x_persist, maxlen=max_len)
        # Build model
        self.keras_model = Sequential()
        embedding_matrix = word_vectors.load_vectors(tokenizer.word_index, language=language)
        self.keras_model.add(
            Embedding(num_words, 300, weights=[embedding_matrix], input_length=max_len, trainable=True))
        self.keras_model.add(Bidirectional(LSTM(recurrent_units)))
        self.keras_model.add(Dropout(dropout))
        self.keras_model.add(Dense(self.output_size, activation='softmax'))

        self.keras_model.compile('adam', 'categorical_crossentropy', metrics=params['metrics'])
