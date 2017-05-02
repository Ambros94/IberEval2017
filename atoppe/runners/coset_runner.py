import csv
from time import gmtime, strftime

from data_loaders import coset
from models.bidirectional_lstm import BidirectionalLSTMModel
from models.cnn import CNNModel
from models.cnn_lstm import CnnLstmModel
from models.fasttext import FastTextModel
from models.lstm import LSTMModel

# Create models
data = coset.load_data(pre_process=True)

b_lstm = BidirectionalLSTMModel(data=data)
b_lstm_f1_micro = b_lstm.run(metrics=['categorical_accuracy', coset.fbeta_score], max_features=13500, max_len=50,
                             batch_size=32, embedding_dims=128, recurrent_units=64, dropout=0.1,
                             epochs=5)

fast_text = FastTextModel(data=data)
fast_text_f1_micro = fast_text.run(metrics=[coset.fbeta_score],
                                   max_features=9300, maxlen=50,
                                   ngram_range=1, embedding_dims=300, hidden_dims=100,
                                   batch_size=32, epochs=5)
lstm = LSTMModel(data=data)
lstm_f1_micro = lstm.run(metrics=[coset.fbeta_score], max_features=13500, maxlen=50, embedding_dims=100, batch_size=32,
                         dropout=0.2, recurrent_dropout=0.4, lstm_units=128, epochs=4)

cnn_lstm = CnnLstmModel(data=data)
cnn_lstm_f1_micro = cnn_lstm.run(metrics=[coset.fbeta_score], max_features=13500, maxlen=50,
                                 embedding_size=128, kernel_size=5, dropout=0.25, strides=1,
                                 filters=64, pool_size=4, lstm_output_size=70, batch_size=30, epochs=3)

cnn = CNNModel(data=data)
cnn_f1_micro = cnn.run(metrics=[coset.fbeta_score], max_features=13500, maxlen=300,
                       batch_size=32, strides=1,
                       embedding_dims=50, filters=100,
                       kernel_size=3, dropout=0.2,
                       dropout_final=0.2,
                       hidden_dims=50, epochs=4)

with open("../coset-" + strftime("%Y%m%d_%H%M%S", gmtime()) + ".log", 'w') as outcsv:
    writer = csv.writer(outcsv, delimiter='\t')
    writer.writerow(["model_name", "test_f1_micro", "test_f1_macro"])
    writer.writerow(["cnn", cnn_f1_micro[0], cnn_f1_micro[1]])
    writer.writerow(["fast_text", fast_text_f1_micro[0], fast_text_f1_micro[1]])
    writer.writerow(["cnn_lstm", cnn_lstm_f1_micro[0], cnn_lstm_f1_micro[1]])
    writer.writerow(["b_lstm", b_lstm_f1_micro[0], b_lstm_f1_micro[1]])
    writer.writerow(["lstm", lstm_f1_micro[0], lstm_f1_micro[1]])
