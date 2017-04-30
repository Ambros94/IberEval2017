from random import choice

from hyperas import optim
from hyperopt import Trials, tpe, STATUS_OK
from keras.layers import Dense, LSTM
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.metrics import f1_score

from data_loaders import coset


def data():
    """
    Provide data for the model in the right format
    :return: x_train, y_train, x_test, y_test
    """
    (ids_train, x_train, y_train), (ids_val, x_val, y_val), (
        ids_test, x_test, y_test) = coset.load_data()
    x_train = sequence.pad_sequences(x_train, maxlen=50)
    x_test = sequence.pad_sequences(x_test, maxlen=50)
    global g_x_test, g_y_test
    g_x_test = x_test
    g_y_test = y_test
    return x_train, y_train, x_test, y_test


def model(x_train, y_train, x_test, y_test):
    # Fixed params

    max_features = 15000
    embedding_dims = {{choice([30, 50, 80, 100, 120, 150, 200, 250, 300])}}
    batch_size = {{choice([16, 32, 64])}}
    dropout = {{choice([0.2, 0.3, 0.4, 0.5])}}
    recurrent_dropout = {{choice([0.2, 0.3, 0.4, 0.5])}}
    epochs = {{choice([3, 4, 5, 6, 7, 8, 9, 10, 11])}}
    units = {{choice([32, 64, 128, 150, 256])}}

    model = Sequential()
    model.add(Embedding(max_features, embedding_dims))
    model.add(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout))
    model.add(Dense(5, activation='softmax'))
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
    print("embedding_dims {}\n".format(embedding_dims))
    print("batch_size {}\n".format(batch_size))
    print("dropout {}\n".format(dropout))
    print("recurrent_dropout {}\n".format(recurrent_dropout))
    print("epochs {}\n".format(epochs))
    print("units {}\n".format(units))
    print("*************\n")

    with open('optimization_result.txt', 'a') as out_file:
        out_file.write("*************\n")
        out_file.write("f1_macro: {f1_macro}\n".format(f1_macro=f1))
        out_file.write("embedding_dims {}\n".format(embedding_dims))
        out_file.write("batch_size {}\n".format(batch_size))
        out_file.write("dropout {}\n".format(dropout))
        out_file.write("recurrent_dropout {}\n".format(recurrent_dropout))
        out_file.write("epochs {}\n".format(epochs))
        out_file.write("units {}\n".format(units))
        out_file.write("*************\n")
    return {'loss': -f1, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    max_evaluations = 2000
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=max_evaluations,
                                          trials=Trials())
    with open('optimization_result.txt', 'a') as out_file:
        out_file.write("Model: {model}".format(model=best_model))
        out_file.write("Parameters: {params}".format(params=best_run))
