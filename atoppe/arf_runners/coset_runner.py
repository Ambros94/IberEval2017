import csv
import inspect
from time import gmtime, strftime

from data_loaders import coset
from deep_models import metrics
from deep_models.bidirectional_lstm import BidirectionalLSTMModel
from deep_models.cnn import CNNModel
from deep_models.fasttext import FastTextModel
from deep_models.kim import KimModel
from deep_models.lstm import LSTMModel
from nlp_utils import tweets_preprocessor


def run():
    data_function = coset.load_data
    max_len = 30
    language = 'es'

    """
    Convolutional Neural Network
    """
    cnn = CNNModel(data_function=data_function, decode_function=coset.decode_label,
                   persist_function=coset.persist_solution,
                   test_function=coset.load_test)
    cnn.run(metrics=[metrics.fbeta_score], maxlen=max_len,
            batch_size=32, strides=1,
            embedding_dims=50, filters=100,
            kernel_size=50, dropout=0.2,
            dropout_final=0.2, dilation_rate=1, padding='same',
            hidden_dims=50, epochs=3, language=language)
    cnn_f1_macro = cnn.test_f1_macro()

    fast_text = FastTextModel(data_function=data_function, decode_function=coset.decode_label,
                              persist_function=coset.persist_solution,
                              test_function=coset.load_test)
    fast_text.run(metrics=[metrics.fbeta_score], maxlen=max_len,
                  ngram_range=2, hidden_dims=128, language=language, noise=0.2,
                  batch_size=32, epochs=8)
    fast_text_f1_macro = fast_text.test_f1_macro()

    lstm = LSTMModel(data_function=data_function, decode_function=coset.decode_label,
                     persist_function=coset.persist_solution,
                     test_function=coset.load_test)
    lstm.run(metrics=[metrics.fbeta_score], maxlen=max_len, language=language, batch_size=32,
             dropout=0.2, recurrent_dropout=0.4, lstm_units=128, epochs=3)
    lstm_f1_macro = lstm.test_f1_macro()

    kim = KimModel(data_function=data_function, decode_function=coset.decode_label,
                   persist_function=coset.persist_solution,
                   test_function=coset.load_test)
    kim.run(metrics=[metrics.fbeta_score], maxlen=max_len,
            batch_size=32, strides=1, filters=150, language=language,
            dropout=0.5, trainable=False,
            epochs=4, padding='same', dilation_rate=4, pool_size=8)
    kim_f1_macro = kim.test_f1_macro()
    b_lstm = BidirectionalLSTMModel(data_function=data_function, decode_function=coset.decode_label,
                                    persist_function=coset.persist_solution,
                                    test_function=coset.load_test)
    b_lstm.run(metrics=[metrics.fbeta_score], max_len=max_len,
               batch_size=32, embedding_dims=128, recurrent_units=64, dropout=0.1, language=language,
               epochs=2)
    b_lstm_f1_macro = b_lstm.test_f1_macro()

    with open("../coset-" + strftime("%Y%m%d_%H%M%S", gmtime()) + ".log", 'w') as outcsv:
        writer = csv.writer(outcsv, delimiter=';')
        writer.writerow(["model_name", "test_f1_macro"])
        writer.writerow(["cnn", ('%.3f' % cnn_f1_macro).replace('.', ',')])
        writer.writerow(["fast_text", ('%.3f' % fast_text_f1_macro).replace('.', ',')])
        writer.writerow(["b_lstm", ('%.3f' % b_lstm_f1_macro).replace('.', ',')])
        writer.writerow(["lstm", ('%.3f' % lstm_f1_macro).replace('.', ',')])
        writer.writerow(["kim", ('%.3f' % kim_f1_macro).replace('.', ',')])
        outcsv.write("Pre-processing:")
        outcsv.write(''.join(inspect.getsourcelines(tweets_preprocessor._clean_tweet)[0]))


run()
