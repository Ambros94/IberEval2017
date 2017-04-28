import logging
from time import gmtime, strftime

from data_loaders import coset
from models.lstm import LSTMModel

logging.basicConfig(filename="../coset-" + strftime("%Y%m%d_%H%M%S", gmtime()) + ".log", level=logging.INFO)

# Create models
coset_data = coset.load_data()

lstm = LSTMModel(data=coset_data)
lstm_acc = lstm.run(metrics=['categorical_accuracy', coset.fbeta_score], max_features=15000, maxlen=50, batch_size=32,
                    epochs=3)
