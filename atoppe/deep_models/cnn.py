from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from deep_models.toppemodel import ToppeModel
from nlp_utils import word_vecors
from nlp_utils.tweets_preprocessor import clean_tweets


class CNNModel(ToppeModel):
    def build(self, params):
        # Params
        language = params['language']
        maxlen = params['maxlen']
        dropout = params['dropout']
        filters = params['filters']
        padding = params['padding']
        kernel_size = params['kernel_size']
        dilation_rate = params['dilation_rate']
        strides = params['strides']
        hidden_dims = params['hidden_dims']
        dropout_final = params['dropout_final']
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
        # Build model
        self.keras_model = Sequential()
        embedding_matrix = word_vecors.load_vectors(tokenizer.word_index, language=language)
        self.keras_model.add(
            Embedding(num_words, 300, weights=[embedding_matrix], input_length=maxlen, trainable=True))
        self.keras_model.add(Dropout(dropout))
        self.keras_model.add(Conv1D(filters,
                                    kernel_size,
                                    padding=padding,
                                    activation='relu',
                                    dilation_rate=dilation_rate,
                                    strides=strides))
        self.keras_model.add(GlobalMaxPooling1D())
        self.keras_model.add(Dense(hidden_dims))
        self.keras_model.add(Dropout(dropout_final))
        self.keras_model.add(Activation('relu'))
        self.keras_model.add(Dense(self.output_size))
        self.keras_model.add(Activation('softmax'))
        # Compile
        self.keras_model.compile(loss='categorical_crossentropy',
                                 optimizer='adam', metrics=params['metrics'])
