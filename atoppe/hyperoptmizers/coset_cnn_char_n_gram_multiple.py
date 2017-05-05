from random import choice

from hyperas import optim
from hyperopt import Trials, tpe, STATUS_OK
from keras.layers import Dense, Activation, Convolution1D, MaxPooling1D, Flatten, Merge
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.metrics import f1_score

from data_loaders import coset
from nlp_utils.n_grams import augment_with_n_grams


def data():
    """
    Provide data for the model in the right format
    :return: x_train, y_train, x_test, y_test
    """
    (ids_train, x_train, y_train), (
        ids_test, x_test, y_test) = coset.load_data(pre_process=True, char_level=True)
    x_train, x_test, max_features = augment_with_n_grams(x_train=x_train, x_test=x_test,
                                                         max_features=160,
                                                         ngram_range=3)
    print(max_features)
    x_train = sequence.pad_sequences(x_train, maxlen=400)
    x_test = sequence.pad_sequences(x_test, maxlen=400)
    return x_train, y_train, x_test, y_test


def model(x_train, y_train, x_test, y_test):
    # Fixed params

    max_len = 400
    # Parameter that need to be optimized

    max_features = 16846
    batch_size = {{choice([16, 32, 64])}}
    embedding_dims = {{choice([25, 50, 75, 100, 125])}}
    filters = {{choice([50, 100, 150, 200, 250, 300])}}
    pooling_length = {{choice([1, 2, 3])}}
    epochs = {{choice([2, 3, 4])}}
    subsample_length = {{choice([1, 2, 3])}}

    branches = []
    for filter_len in [2, 3, 4]:
        branch = Sequential()
        branch.add(Embedding(max_features,
                             embedding_dims,
                             input_length=max_len))
        branch.add(Convolution1D(nb_filter=filters,
                                 filter_length=filter_len,
                                 border_mode='valid',
                                 activation='relu',
                                 subsample_length=subsample_length))
        branch.add(MaxPooling1D(pool_length=pooling_length))
        branch.add(Flatten())

        branches.append(branch)
    model = Sequential()
    model.add(Merge(branches, mode='concat'))

    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[coset.fbeta_score])
    model.fit([x_train, x_train, x_train], y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=([x_test, x_test, x_test], y_test))

    predictions = model.predict([x_test, x_test, x_test], batch_size=batch_size)
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
