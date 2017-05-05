from data_loaders import coset
from models.fasttext import FastTextModel

max_words = 9000
max_len = 30

data = coset.load_data(pre_process=True, use_nltk=True)

fast_text = FastTextModel(data=data)
fast_text_f1_micro = fast_text.run(metrics=[coset.fbeta_score],
                                   max_words=max_words, maxlen=max_len,
                                   ngram_range=2, embedding_dims=300, hidden_dims=50,
                                   batch_size=32, epochs=3)

print(fast_text_f1_micro)
