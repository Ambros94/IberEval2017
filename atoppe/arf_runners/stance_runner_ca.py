import csv
import inspect
from time import gmtime, strftime

from data_loaders import stance
from deep_models import metrics
from deep_models.bidirectional_lstm import BidirectionalLSTMModel
from deep_models.cnn import CNNModel
from deep_models.cnn_lstm import CnnLstmModel
from deep_models.fasttext import FastTextModel
from deep_models.kim import KimModel
from deep_models.lstm import LSTMModel


def run(cleaning_function):
    data_function = stance.load_stance_ca
    test_function = stance.load_test_ca
    max_len = 30
    language = 'ca'

    lstm = LSTMModel(data_function=data_function, decode_function=stance.decode_stance,
                     persist_function=None, test_function=test_function)
    lstm.run(metrics=[metrics.fbeta_score], maxlen=max_len, language=language,
             batch_size=32,
             dropout=0.2, recurrent_dropout=0.4, lstm_units=128, epochs=6, clean_tweets=cleaning_function)
    lstm_metric = lstm.test_f1_macro()

    cnn = CNNModel(data_function=data_function, decode_function=stance.decode_stance,
                   persist_function=None, test_function=test_function)
    cnn.run(metrics=[metrics.fbeta_score], maxlen=max_len,
            batch_size=32, strides=1, filters=100,
            kernel_size=3, dropout=0.2,
            dropout_final=0.2, dilation_rate=1, padding='same',
            hidden_dims=50, epochs=2, language=language, clean_tweets=cleaning_function)
    cnn_accuracy = cnn.test_f1_macro()

    b_lstm = BidirectionalLSTMModel(data_function=data_function, decode_function=stance.decode_stance,
                                    persist_function=None, test_function=test_function)
    b_lstm.run(metrics=[metrics.fbeta_score], max_len=max_len,
               batch_size=32, recurrent_units=64, dropout=0.1, language=language,
               epochs=2, clean_tweets=cleaning_function)
    b_lstm_accuracy = b_lstm.test_f1_macro()

    cnn_lstm = CnnLstmModel(data_function=data_function, decode_function=stance.decode_stance,
                            persist_function=None, test_function=test_function)
    cnn_lstm.run(metrics=[metrics.fbeta_score], maxlen=max_len, kernel_size=5, dropout=0.25, strides=1,
                 language=language,
                 filters=64, pool_size=4, lstm_output_size=70, batch_size=30, epochs=4, clean_tweets=cleaning_function)
    cnn_lstm_accuracy = cnn_lstm.test_f1_macro()

    fast_text = FastTextModel(data_function=data_function, decode_function=stance.decode_stance,
                              persist_function=None, test_function=stance.load_test_ca)
    fast_text.run(metrics=[metrics.fbeta_score], maxlen=max_len,
                  ngram_range=2, hidden_dims=100, language=language,
                  batch_size=32, epochs=3, noise=0.2, clean_tweets=cleaning_function)
    fast_text_accuracy = fast_text.test_f1_macro()

    kim = KimModel(data_function=data_function, decode_function=stance.decode_stance,
                   persist_function=None, test_function=stance.load_test_ca)
    kim.run(metrics=[metrics.fbeta_score], maxlen=max_len,
            batch_size=32, strides=1, filters=150, language=language,
            dropout=0.5, dropout_final=0.5, trainable=True,
            recurrent_units=128, epochs=4, padding='same', dilation_rate=3, pool_size=5, clean_tweets=cleaning_function)
    kim_accuracy = kim.test_f1_macro()

    with open("../stance_ca-" + strftime("%Y%m%d_%H%M%S", gmtime()) + ".log", 'w') as outcsv:
        writer = csv.writer(outcsv, delimiter='\t')
        writer.writerow(["model_name", "test_accuracy"])
        writer.writerow(["cnn", cnn_accuracy])
        writer.writerow(["fast_text", fast_text_accuracy])
        writer.writerow(["cnn_lstm", cnn_lstm_accuracy])
        writer.writerow(["b_lstm", b_lstm_accuracy])
        writer.writerow(["lstm", lstm_metric])
        writer.writerow(["kim", kim_accuracy])
        print(["model_name", "test_accuracy"])
        print(["cnn", cnn_accuracy])
        print(["fast_text", fast_text_accuracy])
        print(["cnn_lstm", cnn_lstm_accuracy])
        print(["b_lstm", b_lstm_accuracy])
        print(["lstm", lstm_metric])
        print(["kim", kim_accuracy])
        outcsv.write("Pre-processing:")
        outcsv.write(''.join(inspect.getsourcelines(cleaning_function)[0]))
