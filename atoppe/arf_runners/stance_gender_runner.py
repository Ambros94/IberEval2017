import csv
import inspect
from time import gmtime, strftime

from data_loaders import stance
from deep_models.bidirectional_lstm import BidirectionalLSTMModel
from deep_models.cnn import CNNModel
from deep_models.cnn_lstm import CnnLstmModel
from deep_models.fasttext import FastTextModel
from deep_models.kim import KimModel
from deep_models.lstm import LSTMModel
from nlp_utils import tweets_preprocessor

# Create models
data_function = stance.load_gender_ca
max_len = 30
language = 'ca'

cnn = CNNModel(data_function=data_function, decode_function=stance.decode_gender,
               persist_function=stance.persist_gender)
cnn.run(metrics=['categorical_accuracy'], maxlen=max_len,
        batch_size=32, strides=1,
        embedding_dims=50, filters=100,
        kernel_size=3, dropout=0.2,
        dropout_final=0.2, dilation_rate=1, padding='same',
        hidden_dims=50, epochs=2, language=language)
cnn_accuracy = cnn.test_accuracy()

b_lstm = BidirectionalLSTMModel(data_function=data_function, decode_function=stance.decode_gender,
                                persist_function=stance.persist_gender)
b_lstm.run(metrics=['categorical_accuracy'], max_len=max_len,
           batch_size=32, embedding_dims=128, recurrent_units=64, dropout=0.1, language=language,
           epochs=2)
b_lstm_accuracy = b_lstm.test_accuracy()

lstm = LSTMModel(data_function=data_function, decode_function=stance.decode_gender,
                 persist_function=stance.persist_gender)
lstm.run(metrics=['categorical_accuracy'], maxlen=max_len, embedding_dims=100, language=language,
         batch_size=32,
         dropout=0.2, recurrent_dropout=0.4, lstm_units=128, epochs=4)
lstm_accuracy = lstm.test_accuracy()

cnn_lstm = CnnLstmModel(data_function=data_function, decode_function=stance.decode_gender,
                        persist_function=stance.persist_gender)
cnn_lstm.run(metrics=['categorical_accuracy'], maxlen=max_len,
             embedding_size=128, kernel_size=5, dropout=0.25, strides=1, language=language,
             filters=64, pool_size=4, lstm_output_size=70, batch_size=30, epochs=2)
cnn_lstm_accuracy = cnn_lstm.test_accuracy()

fast_text = FastTextModel(data_function=data_function, decode_function=stance.decode_gender,
                          persist_function=stance.persist_gender)
fast_text.run(metrics=['categorical_accuracy'], maxlen=max_len,
              ngram_range=2, embedding_dims=300, hidden_dims=100, language=language,
              batch_size=32, epochs=6)
fast_text_accuracy = fast_text.test_accuracy()

kim = KimModel(data_function=data_function, decode_function=stance.decode_gender,
               persist_function=stance.persist_gender)
kim.run(metrics=['categorical_accuracy'], maxlen=max_len,
        batch_size=32, strides=1, embedding_dims=150, filters=150, language=language,
        dropout=0.5, dropout_final=0.5, trainable=True,
        recurrent_units=128, epochs=3, padding='same', dilation_rate=3, pool_size=5)
kim_accuracy = kim.test_accuracy()

with open("../coset-" + strftime("%Y%m%d_%H%M%S", gmtime()) + ".log", 'w') as outcsv:
    writer = csv.writer(outcsv, delimiter='\t')
    writer.writerow(["model_name", "test_accuracy"])
    writer.writerow(["cnn", cnn_accuracy])
    writer.writerow(["fast_text", fast_text_accuracy])
    writer.writerow(["cnn_lstm", cnn_lstm_accuracy])
    writer.writerow(["b_lstm", b_lstm_accuracy])
    writer.writerow(["lstm", lstm_accuracy])
    writer.writerow(["kim", kim_accuracy])
    outcsv.write("Pre-processing:")
    outcsv.write(''.join(inspect.getsourcelines(tweets_preprocessor._clean_tweet)[0]))
