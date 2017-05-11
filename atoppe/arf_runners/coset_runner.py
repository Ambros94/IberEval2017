from data_loaders import coset
from deep_models import metrics
from deep_models.fasttext import FastTextModel

data_function = coset.load_data
test_ids, test_data = coset.load_test()
max_len = 50

fast_text = FastTextModel(data_function=data_function, decode_function=coset.decode_label,
                          persist_function=coset.persist_solution,
                          test_function=coset.load_test)
fast_text.run(metrics=[metrics.fbeta_score], maxlen=max_len,
              ngram_range=2, embedding_dims=300, hidden_dims=256, language=language,
              batch_size=32, epochs=7)
print(fast_text.test_f1_macro())

"""

NOTES:

- Nadam - 512 neurons
    - 6 epochs: 0.57
    - 7 epochs: 0.593394318859
    - 8 epochs: 0.568823251525
    - 9 epochs: 0.568974516636
"""
