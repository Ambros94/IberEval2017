import csv
import inspect
from time import gmtime, strftime

from data_loaders import coset
from deep_models import metrics
from deep_models.bidirectional_lstm import BidirectionalLSTMModel
from deep_models.cnn import CNNModel
from deep_models.cnn_lstm import CnnLstmModel
from deep_models.fasttext import FastTextModel
from deep_models.kim import KimModel
from deep_models.lstm import LSTMModel
from nlp_utils import tweets_preprocessor

# Create models
data = coset.load_data()
(ids_train, x_train, y_train), (ids_test, x_test, y_test) = data
max_len = 30
language = 'es'

cnn = CNNModel(data=data)
cnn.run(metrics=[metrics.fbeta_score], maxlen=max_len,
        batch_size=32, strides=1,
        embedding_dims=50, filters=100,
        kernel_size=50, dropout=0.2,
        dropout_final=0.2, dilation_rate=1, padding='same',
        hidden_dims=50, epochs=3, language=language)
cnn_f1_macro = cnn.test_f1_macro()

b_lstm = BidirectionalLSTMModel(data=data)
b_lstm.run(metrics=[metrics.fbeta_score], max_len=max_len,
           batch_size=32, embedding_dims=128, recurrent_units=64, dropout=0.1, language=language,
           epochs=2)
b_lstm_f1_macro = b_lstm.test_f1_macro()

lstm = LSTMModel(data=data)
lstm.run(metrics=[metrics.fbeta_score], maxlen=max_len, embedding_dims=100, language=language,
         batch_size=32,
         dropout=0.2, recurrent_dropout=0.4, lstm_units=128, epochs=4)
lstm_f1_macro = lstm.test_f1_macro()

cnn_lstm = CnnLstmModel(data=data)
cnn_lstm.run(metrics=[metrics.fbeta_score], maxlen=max_len,
             embedding_size=128, kernel_size=5, dropout=0.25, strides=1, language=language,
             filters=64, pool_size=4, lstm_output_size=70, batch_size=30, epochs=2)
cnn_lstm_f1_macro = cnn_lstm.test_f1_macro()

fast_text = FastTextModel(data=data)
fast_text.run(metrics=[metrics.fbeta_score], maxlen=max_len,
              ngram_range=2, embedding_dims=300, hidden_dims=100, language=language,
              batch_size=32, epochs=6)
fast_text_f1_macro = fast_text.test_f1_macro()

kim = KimModel(data=data)
kim.run(metrics=[metrics.fbeta_score], maxlen=max_len,
        batch_size=32, strides=1, embedding_dims=150, filters=150, language=language,
        dropout=0.5, dropout_final=0.5, trainable=True,
        recurrent_units=128, epochs=3, padding='same', dilation_rate=3, pool_size=5)
kim_f1_macro = kim.test_f1_macro()

with open("../coset-" + strftime("%Y%m%d_%H%M%S", gmtime()) + ".log", 'w') as outcsv:
    writer = csv.writer(outcsv, delimiter='\t')
    writer.writerow(["model_name", "test_f1_macro", "test_f1_macro"])
    writer.writerow(["cnn", cnn_f1_macro])
    writer.writerow(["fast_text", fast_text_f1_macro])
    writer.writerow(["cnn_lstm", cnn_lstm_f1_macro])
    writer.writerow(["b_lstm", b_lstm_f1_macro])
    writer.writerow(["lstm", lstm_f1_macro])
    writer.writerow(["kim", kim_f1_macro])
    outcsv.write("Pre-processing:")
    outcsv.write(''.join(inspect.getsourcelines(tweets_preprocessor._clean_tweet)[0]))
