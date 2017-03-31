from data_loaders import coset
from models.bidirectional_lstm import BidirectionalLSTMModel
from models.cnn import CNNModel
from models.cnn_lstm import CnnLstmModel
from models.fasttext import FastTextModel
from models.lstm import LSTMModel

fast_text = FastTextModel(data_function=coset.load_data)
fast_text_acc = fast_text.run(metrics=[coset.fbeta_score, coset.precision, coset.recall],
                              max_features=15000, maxlen=50,
                              ngram_range=1, embedding_dims=50,
                              batch_size=32, epochs=15)

cnn = CNNModel(data_function=coset.load_data)
cnn_acc = cnn.run(metrics=[coset.fbeta_score, coset.precision, coset.recall], max_features=15000, maxlen=50,
                  batch_size=32,
                  embedding_dims=50, filters=250,
                  kernel_size=3,
                  hidden_dims=250, epochs=5)

b_cnn_lstm = CnnLstmModel(data_function=coset.load_data)
b_cnn_lstm_acc = b_cnn_lstm.run(metrics=[coset.fbeta_score], max_features=15000, maxlen=50,
                                embedding_size=128, kernel_size=5,
                                filters=64, pool_size=4, lstm_output_size=70, batch_size=30, epochs=6)

b_lstm = BidirectionalLSTMModel(data_function=coset.load_data)
b_lstm_acc = b_lstm.run(metrics=[coset.fbeta_score], max_features=15000, maxlen=50, batch_size=32,
                        epochs=3)

lstm = LSTMModel(data_function=coset.load_data)
lstm_acc = lstm.run(metrics=[coset.fbeta_score], max_features=15000, maxlen=50, batch_size=32, epochs=3)

print('')
print('CNN accuracy:\t', cnn_acc)
print('BidirectionalLSTM accuracy:\t', b_lstm_acc)
print('CNN_LSTM accuracy:\t', b_cnn_lstm_acc)
print('LSTM accuracy:\t', lstm_acc)
print('FastText accuracy:\t', fast_text_acc)
