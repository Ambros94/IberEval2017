from data_loaders import coset
from deep_models import metrics
from deep_models.lstm import LSTMModel

max_len = 30

data = coset.load_data()

lstm = LSTMModel(data=data)
lstm_f1_micro = lstm.run(metrics=[metrics.fbeta_score], maxlen=max_len, embedding_dims=100,
                         batch_size=32,
                         dropout=0.2, recurrent_dropout=0.4, lstm_units=128, epochs=4)

print(lstm_f1_micro)
