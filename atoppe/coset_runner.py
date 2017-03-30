from data_loaders import coset
from models.bidirectional_lstm import BidirectionalLSTMModel
from models.cnn import CNNModel
from models.cnn_lstm import CnnLstmModel
from models.fasttext import FastTextModel
from models.lstm import LSTMModel

b_cnn_lstm = CnnLstmModel(data_function=coset.load_data)
b_cnn_lstm_score, b_cnn_lstm_acc = b_cnn_lstm.run(max_features=15000, maxlen=50, embedding_size=128, kernel_size=5,
                                                  filters=64, pool_size=4, lstm_output_size=70, batch_size=30, epochs=6)
print('CNN_LSTM accuracy:', b_cnn_lstm)

cnn_lstm = CnnLstmModel(data_function=coset.load_data)
cnn_lstm_score, cnn_lstm_acc = cnn_lstm.run(max_features=15000, maxlen=50, batch_size=32, epochs=3)
print('BidirectionalLSTM accuracy:', cnn_lstm)

b_lstm = BidirectionalLSTMModel(data_function=coset.load_data)
b_lstm_score, b_lstm_acc = b_lstm.run(max_features=15000, maxlen=50, batch_size=32, epochs=3)
print('BidirectionalLSTM accuracy:', b_lstm_acc)

fast_text = FastTextModel(data_function=coset.load_data)
fast_text_score, fast_text_acc = fast_text.run(max_features=15000, maxlen=50, ngram_range=3, embedding_dims=50,
                                               batch_size=32, epochs=3)
print('FastText accuracy:', fast_text_acc)

lstm = LSTMModel(data_function=coset.load_data)
lstm_score, lstm_acc = lstm.run(max_features=15000, maxlen=50, batch_size=32, epochs=3)
print('LSTM accuracy:', lstm_acc)

cnn = CNNModel(data_function=coset.load_data)
cnn_score, cnn_acc = cnn.run(max_features=15000, maxlen=50, batch_size=32, embedding_dims=50, filters=250,
                             kernel_size=3,
                             hidden_dims=250, epochs=5)
print('CNN accuracy:', cnn_acc)
