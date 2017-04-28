import logging
from time import gmtime, strftime

from data_loaders import coset
from models.bidirectional_lstm import BidirectionalLSTMModel
from models.cnn import CNNModel
from models.cnn_lstm import CnnLstmModel
from models.fasttext import FastTextModel
from models.lstm import LSTMModel

logging.basicConfig(filename="../coset-" + strftime("%Y%m%d_%H%M%S", gmtime()) + ".log", level=logging.INFO)

# Create models
data = coset.load_data()

fast_text = FastTextModel(data=data)
fast_text_acc = fast_text.run(metrics=['categorical_accuracy'],
                              max_features=15000, maxlen=400,
                              ngram_range=3, embedding_dims=50,
                              batch_size=64, epochs=50)

cnn = CNNModel(data=data)
cnn_acc = cnn.run(metrics=['categorical_accuracy'], max_features=15000, maxlen=50,
                  batch_size=32,
                  embedding_dims=50, filters=250,
                  kernel_size=3,
                  hidden_dims=250, epochs=5)

cnn_lstm = CnnLstmModel(data=data)
cnn_lstm_acc = cnn_lstm.run(metrics=['categorical_accuracy'], max_features=15000, maxlen=50,
                                embedding_size=128, kernel_size=5,
                                filters=64, pool_size=4, lstm_output_size=70, batch_size=30, epochs=6)

b_lstm = BidirectionalLSTMModel(data=data)
b_lstm_acc = b_lstm.run(metrics=['categorical_accuracy'], max_features=15000, maxlen=50, batch_size=32,
                        epochs=3)

lstm = LSTMModel(data=data)
lstm_acc = lstm.run(metrics=['categorical_accuracy'], max_features=15000, maxlen=50, batch_size=32, epochs=3)

# Log everything on file
logging.info('CNN accuracy:\t{f1_score}'.format(f1_score=cnn_acc))
logging.info('BidirectionalLSTM accuracy:\t{f1_score}'.format(f1_score=b_lstm_acc))
logging.info('CNN_LSTM accuracy:\t{f1_score}'.format(f1_score=cnn_lstm_acc))
logging.info('LSTM accuracy:\t{f1_score}'.format(f1_score=lstm_acc))
logging.info('FastText accuracy:\t{f1_score}'.format(f1_score=fast_text_acc))
