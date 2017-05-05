from data_loaders import coset
from models import metrics
from models.fasttext import FastTextModel

max_len = 30

data = coset.load_data()

fast_text = FastTextModel(data=data)
fast_text_f1_micro = fast_text.run(metrics=[metrics.fbeta_score], maxlen=max_len,
                                   ngram_range=2, embedding_dims=300, hidden_dims=50,
                                   batch_size=32, epochs=3)

print(fast_text_f1_micro)
