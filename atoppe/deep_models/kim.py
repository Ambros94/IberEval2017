import keras
from keras.engine import Input, Model
from keras.layers import Conv1D, Dense, Flatten, GaussianNoise, Activation
from keras.layers import Embedding
from keras.layers import MaxPooling1D
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from deep_models.toppemodel import ToppeModel
from nlp_utils import word_vectors
from nlp_utils.tweets_preprocessor import clean_tweets


class KimModel(ToppeModel):
    def build(self, params):
        # Cleaning data
        language = params['language']
        clean_function = params['clean_tweets']
        self.x_train = clean_tweets(cleaning_function=clean_function, tweets=self.x_train)
        self.x_test = clean_tweets(cleaning_function=clean_function, tweets=self.x_test)
        self.x_persist = clean_tweets(cleaning_function=clean_function, tweets=self.x_persist)
        # Prepare data
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.x_train)
        num_words = len(tokenizer.word_index) + 1
        x_train = tokenizer.texts_to_sequences(self.x_train)
        x_test = tokenizer.texts_to_sequences(self.x_test)
        x_persist = tokenizer.texts_to_sequences(self.x_persist)
        self.x_train = sequence.pad_sequences(x_train, maxlen=params['maxlen'])
        self.x_test = sequence.pad_sequences(x_test, maxlen=params['maxlen'])
        self.x_persist = sequence.pad_sequences(x_persist, maxlen=params['maxlen'])

        embedding_matrix = word_vectors.load_vectors(tokenizer.word_index, language=language)
        # Create real model
        x = Input(shape=(params['maxlen'],))
        emb = Embedding(num_words,
                        300, weights=[embedding_matrix], trainable=True,
                        input_length=params['maxlen'])(x)
        noisy_embedding = GaussianNoise(0.2)(emb)
        # First Branch
        a_conv = Conv1D(filters=params['filters'],
                        kernel_size=2,
                        padding='same',
                        dilation_rate=1,
                        input_shape=(params['maxlen'], 300))(noisy_embedding)
        a_activation = Activation('relu')(a_conv)
        a_max_pooling = MaxPooling1D(pool_size=5)(a_activation)
        a_flatten = Flatten()(a_max_pooling)
        # Second Branch
        b_conv = Conv1D(filters=params['filters'],
                        kernel_size=2,
                        padding='valid',
                        input_shape=(params['maxlen'], 300))(noisy_embedding)
        b_activation = Activation('relu')(b_conv)
        b_max_pooling = MaxPooling1D(pool_size=5)(b_activation)
        b_flatten = Flatten()(b_max_pooling)
        # First Branch
        c_conv = Conv1D(filters=params['filters'],
                        kernel_size=3,
                        padding='same',
                        dilation_rate=1,
                        input_shape=(params['maxlen'], 300))(noisy_embedding)
        c_activation = Activation('relu')(c_conv)
        c_max_pooling = MaxPooling1D(pool_size=5)(c_activation)
        c_flatten = Flatten()(c_max_pooling)
        # First Branch
        d_conv = Conv1D(filters=params['filters'],
                        kernel_size=7,
                        padding=params['padding'],
                        dilation_rate=1,
                        input_shape=(params['maxlen'], 300))(noisy_embedding)
        d_activation = Activation('relu')(d_conv)
        d_max_pooling = MaxPooling1D(pool_size=5)(d_activation)
        d_flatten = Flatten()(d_max_pooling)
        # Merge everything together
        merge_input = [a_flatten, b_flatten, c_flatten, d_flatten]
        merged = keras.layers.concatenate(merge_input)
        noisy_merge = GaussianNoise(0.2)(merged)
        y = Dense(self.output_size, activation='sigmoid')(noisy_merge)
        self.keras_model = Model(inputs=x, outputs=y)

        self.keras_model.compile(loss='categorical_crossentropy',
                                 optimizer='adam', metrics=params['metrics'])
