from random import choice

from hyperas import optim
from hyperopt import Trials, tpe, STATUS_OK
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence

import coset


def data():
    """
    Provide data for the model in the right format
    :return: x_train, y_train, x_test, y_test
    """
    (x_train, y_train), (x_test, y_test) = coset.load_data()
    x_train = sequence.pad_sequences(x_train, maxlen=50)
    x_test = sequence.pad_sequences(x_test, maxlen=50)
    return x_train, y_train, x_test, y_test


def model(x_train, y_train, x_test, y_test):
    # Fixed params

    max_len = 50
    # Parameter that need to be optimized

    max_features = 15000
    batch_size = 32
    embedding_dims = {{choice([25, 50, 100])}}
    filters = {{choice([150, 200, 250, 300])}}
    kernel_size = {{choice([2, 3, 4, 5])}}
    hidden_dims = {{choice([150, 200, 250, 300])}}
    epochs = {{choice([5, 6, 7, 8])}}

    model = Sequential()

    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=max_len))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(GlobalMaxPooling1D())

    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(5))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test))

    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    max_evaluations = 100
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=max_evaluations,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Best performing model performances: {performance}".format(performance=best_model.evaluate(X_test, Y_test)))
    print("Best performing model parameters: {params}".format(params=best_run))
