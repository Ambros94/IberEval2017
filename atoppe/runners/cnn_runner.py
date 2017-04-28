import logging
from time import gmtime, strftime

from data_loaders import coset
from models.lstm import LSTMModel

logging.basicConfig(filename="../coset-" + strftime("%Y%m%d_%H%M%S", gmtime()) + ".log", level=logging.INFO)

# Create models
coset_data = coset.load_data()

lstm = LSTMModel(data=coset_data)
lstm_acc = lstm.run(metrics=[coset.fbeta_score], max_features=15000, maxlen=50, batch_size=32,
                    epochs=3)
print(lstm.evaluate_test(batch_size=32))
lstm.persist_result()
