from random import choice

import keras
from hyperas import optim
from hyperopt import Trials, tpe, STATUS_OK
from keras.engine import Input, Model
from keras.layers import Dense, MaxPooling1D, Flatten, Conv1D
from keras.layers import Embedding
from keras.preprocessing import sequence
from sklearn.metrics import f1_score

from data_loaders import coset


def data():
    """
    Provide data for the model in the right format
    :return: x_train, y_train, x_test, y_test
    """
    (ids_train, x_train, y_train), (
        ids_test, x_test, y_test) = coset.load_data(pre_process=True, char_level=False)
    x_train = sequence.pad_sequences(x_train, maxlen=30)
    x_test = sequence.pad_sequences(x_test, maxlen=30)
    return x_train, y_train, x_test, y_test


def model(x_train, y_train, x_test, y_test):
    # Fixed params

    max_len = 30
    # Parameter that need to be optimized

    max_features = 8100
    batch_size = {{choice([16, 32, 64])}}
    embedding_dims = {{choice([25, 50, 75, 100, 125, 200, 300])}}
    filters = {{choice([50, 100, 150, 200, 250, 300])}}
    pooling_length = {{choice([1, 2, 3])}}
    epochs = {{choice([2, 3,])}}
    padding = {{choice(["valid", "same"])}}
    dilation = {{choice([2, 3, 4])}}

    x = Input(shape=(max_len,))
    emb = Embedding(max_features,
                    embedding_dims,
                    input_length=max_len)(x)
    merge_input = []
    for kernel_size in [2, 3, 7, 8]:
        conv = Conv1D(filters=filters,
                      kernel_size=kernel_size,
                      padding=padding,
                      dilation_rate=dilation,
                      activation='relu', input_shape=(max_len, embedding_dims))(emb)
        max_pooling = MaxPooling1D(pool_size=pooling_length)(conv)
        flatten = Flatten()(max_pooling)
        merge_input.append(flatten)

    merged = keras.layers.concatenate(merge_input)
    y = Dense(5, activation='sigmoid')(merged)
    model = Model(inputs=x, outputs=y)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[coset.fbeta_score])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test))

    predictions = model.predict(x_test, batch_size=batch_size)
    f1 = f1_score(coset.decode_labels(y_test), coset.decode_labels(predictions), average='macro')
    print("*************\n")
    print("f1_macro: {f1_macro}\n".format(f1_macro=f1))
    print("*************\n")
    return {'loss': -f1, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    max_evaluations = 200
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=max_evaluations,
                                          trials=Trials())
    print("Parameters: {params}".format(params=best_run))
