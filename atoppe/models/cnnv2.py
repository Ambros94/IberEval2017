import keras
import numpy as np
from keras.engine import Input, Model
from keras.layers import Conv1D, Dense, LSTM
from keras.layers import Embedding
from keras.layers import MaxPooling1D, Dropout
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from models.mymodel import MyModel


class CNNModelv2(MyModel):
    def build(self, params):
        # Prepare data
        tokenizer = Tokenizer(num_words=params['max_words'])
        tokenizer.fit_on_texts(self.x_train)
        x_train = tokenizer.texts_to_sequences(self.x_train)
        x_test = tokenizer.texts_to_sequences(self.x_test)
        print('Found {word_index} unique tokens'.format(word_index=len(tokenizer.word_index)))
        self.x_train = sequence.pad_sequences(x_train, maxlen=params['maxlen'])
        self.x_test = sequence.pad_sequences(x_test, maxlen=params['maxlen'])
        # Prepare embedding
        print('Indexing word vectors.')
        embeddings_index = {}
        f = open("/Users/lambrosini/PycharmProjects/IberEval2017/resources/word2vec/es.vec")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))
        print('Preparing embedding matrix.')

        # prepare embedding matrix
        num_words = min(params['max_words'], len(tokenizer.word_index)) + 1
        embedding_matrix = np.zeros((num_words, 300))
        found, oob = 0, 0
        for word, i in tokenizer.word_index.items():
            if i >= params['max_words']:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
                found += 1
            else:
                oob += 1
        print("Found", found)
        print("OOB", oob)

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
                          activation='relu', input_shape=(params['maxlen'], params['embedding_dims']))(emb)
            drop = Dropout(params['dropout'])(conv)
            max_pooling = MaxPooling1D(pool_size=params['pool_size'])(drop)
            # flatten = Flatten()(max_pooling)
            merge_input.append(max_pooling)

        merged = keras.layers.concatenate(merge_input)
        hidden = LSTM(params['recurrent_units'], recurrent_dropout=0.5, dropout=0.5)(merged)
        y = Dense(self.output_size, activation='sigmoid')(hidden)
        self.keras_model = Model(inputs=x, outputs=y)

        self.keras_model.compile(loss='categorical_crossentropy',
                                 optimizer='adam', metrics=params['metrics'])
