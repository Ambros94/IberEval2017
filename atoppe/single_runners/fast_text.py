from data_loaders import coset
from deep_models import metrics
from deep_models.fasttext import FastTextModel

max_len = 30

data = coset.load_data()

fast_text = FastTextModel(data=data)
fast_text.run(metrics=[metrics.fbeta_score], maxlen=max_len,
                                   ngram_range=2, embedding_dims=300, hidden_dims=50,
                                   batch_size=32, epochs=3)
fast_text_f1_macro = fast_text.test_f1_macro()

print(fast_text_f1_macro)
