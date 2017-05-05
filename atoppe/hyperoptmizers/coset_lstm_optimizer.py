from random import choice

from hyperas import optim
from hyperopt import Trials, tpe, STATUS_OK
from keras.layers import Dense, LSTM
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score

from data_loaders import coset
from deep_models import metrics
from nlp_utils import word_vecors
from nlp_utils.tweets_preprocessor import clean_tweets


def data():
    """
    Provide data for the model in the right format
    :return: x_train, y_train, x_test, y_test
    """
    (ids_train, x_train, y_train), (
        ids_test, x_test, y_test) = coset.load_data()

    return x_train, y_train, x_test, y_test


def model(x_train, y_train, x_test, y_test):
    # Fixed params

    embedding_dims = {{choice([30, 50, 80, 100, 120, 150, 200, 250, 300])}}
    batch_size = {{choice([16, 32, 64])}}
    dropout = {{choice([0.2, 0.3, 0.4, 0.5])}}
    recurrent_dropout = {{choice([0.2, 0.3, 0.4, 0.5])}}
    epochs = {{choice([3, 4, 5, 6, 7, 8])}}
    units = {{choice([32, 64, 128])}}
    units_2 = {{choice([32, 64, 128])}}
    units_3 = {{choice([32, 64, 128])}}
    x_train = clean_tweets(x_train)
    x_test = clean_tweets(x_test)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)
    num_words = len(tokenizer.word_index) + 1
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    print('Found {word_index} words'.format(word_index=num_words))
    x_train = sequence.pad_sequences(x_train, maxlen=30)
    x_test = sequence.pad_sequences(x_test, maxlen=30)

    model = Sequential()
    embedding_matrix = word_vecors.load_vectors(tokenizer.word_index, language='en')
    model.add(Embedding(num_words, 300, weights=[embedding_matrix], input_length=num_words, trainable=True))
    model.add(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))
    model.add(LSTM(units_2, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))
    model.add(LSTM(units_3, dropout=dropout, recurrent_dropout=recurrent_dropout))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[metrics.fbeta_score])

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
    print("units_2 {}\n".format(units_2))
    print("units_3 {}\n".format(units_3))
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
