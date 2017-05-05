from random import choice

import keras
import numpy as np
from hyperas import optim
from hyperopt import Trials, tpe, STATUS_OK
from keras.engine import Input, Model
from keras.layers import Dense, MaxPooling1D, Conv1D, LSTM
from keras.layers import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score

from data_loaders import coset


def data():
    """
    Provide data for the model in the right format
    :return: x_train, y_train, x_test, y_test
    """
    (ids_train, x_train, y_train), (
        ids_test, x_test, y_test) = coset.load_data(pre_process=True, char_level=False)
    return x_train, y_train, x_test, y_test


def model(x_train, y_train, x_test, y_test):
    params = {'max_words': 10000,
              'maxlen': 30,
              'kernel_sizes': 1,
              'filters': 1,
              'dilation_rate': 2,
              'pool_size': 3,
              'recurrent_units': 3
              }
    epochs = {{choice([2, 3,4,5])}}

    tokenizer = Tokenizer(num_words=params['max_words'])
    tokenizer.fit_on_texts(x_train)
    x_train2 = tokenizer.texts_to_sequences(x_train)
    x_test2 = tokenizer.texts_to_sequences(x_test)
    print('Found {word_index} unique tokens'.format(word_index=len(tokenizer.word_index)))
    x_train = sequence.pad_sequences(x_train, maxlen=params['maxlen'])
    x_test = sequence.pad_sequences(x_test, maxlen=params['maxlen'])
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

    num_words = min(params['max_words'], len(tokenizer.word_index)) + 1
    embedding_matrix = np.zeros((num_words, 300))
    for word, i in tokenizer.word_index.items():
        if i >= params['max_words']:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    x = Input(shape=(params['maxlen'],))
    emb = Embedding(num_words,
                    300, weights=[embedding_matrix], trainable=True,
                    input_length=params['maxlen'])(x)

    merge_input = []
    for kernel_size in params['kernel_sizes']:
        conv = Conv1D(filters=params['filters'],
                      kernel_size=kernel_size,
                      padding="valid",
                      dilation_rate=params['dilation_rate'],
                      activation='relu', input_shape=(params['maxlen'], 300))(emb)
        max_pooling = MaxPooling1D(pool_size=params['pool_size'])(conv)
        merge_input.append(max_pooling)

    merged = keras.layers.concatenate(merge_input)
    hidden = LSTM(params['recurrent_units'], recurrent_dropout=0.5, dropout=0.5)(merged)
    y = Dense(5, activation='sigmoid')(hidden)
    keras_model = Model(inputs=x, outputs=y)

    keras_model.compile(loss='categorical_crossentropy',
                        optimizer='adam', metrics=coset.fbeta_score())

    keras_model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_test, y_test))

    predictions = keras_model.predict(x_test, batch_size=32)
    f1 = f1_score(coset.decode_labels(y_test), coset.decode_labels(predictions), average='macro')
    print("*************\n")
    print("f1_macro: {f1_macro}\n".format(f1_macro=f1))
    print("*************\n")
    return {'loss': -f1, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    max_evaluations = 5
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=max_evaluations,
                                          trials=Trials())
    print("Parameters: {params}".format(params=best_run))
