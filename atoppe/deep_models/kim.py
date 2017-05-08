import keras
from keras.engine import Input, Model
from keras.layers import Conv1D, Dense, Flatten
from keras.layers import Embedding
from keras.layers import MaxPooling1D, Dropout
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from deep_models.toppemodel import ToppeModel
from nlp_utils import word_vecors
from nlp_utils.tweets_preprocessor import clean_tweets


class KimModel(ToppeModel):
    def build(self, params):
        # Cleaning data
        language = params['language']
        self.x_train = clean_tweets(self.x_train)
        self.x_test = clean_tweets(self.x_test)
        # Prepare data
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.x_train)
        num_words = len(tokenizer.word_index) + 1
        x_train = tokenizer.texts_to_sequences(self.x_train)
        x_test = tokenizer.texts_to_sequences(self.x_test)
        x_persist = tokenizer.texts_to_sequences(self.x_persist)
        print('Found {word_index} words'.format(word_index=num_words))
        self.x_train = sequence.pad_sequences(x_train, maxlen=params['maxlen'])
        self.x_test = sequence.pad_sequences(x_test, maxlen=params['maxlen'])
        self.x_persist = sequence.pad_sequences(x_persist, maxlen=params['maxlen'])
        embedding_matrix = word_vecors.load_vectors(tokenizer.word_index, language=language)
        # Create real model
        x = Input(shape=(params['maxlen'],))
        emb = Embedding(num_words,
                        300, weights=[embedding_matrix], trainable=params['trainable'],
                        input_length=params['maxlen'])(x)

        merge_input = []
        for kernel_size in [2, 3, 5, 8]:
            conv = Conv1D(filters=params['filters'],
                          kernel_size=kernel_size,
                          padding=params['padding'],
                          dilation_rate=params['dilation_rate'],
                          activation='relu', input_shape=(params['maxlen'], 300))(emb)
            drop = Dropout(params['dropout'])(conv)
            max_pooling = MaxPooling1D(pool_size=params['pool_size'])(drop)
            flatten = Flatten()(max_pooling)
            merge_input.append(flatten)

        merged = keras.layers.concatenate(merge_input)
        y = Dense(self.output_size, activation='sigmoid')(merged)
        self.keras_model = Model(inputs=x, outputs=y)

        self.keras_model.compile(loss='categorical_crossentropy',
                                 optimizer='adam', metrics=params['metrics'])
