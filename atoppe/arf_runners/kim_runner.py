from data_loaders import coset
from deep_models import metrics
from deep_models.kim import KimModel

data_function = coset.load_data
test_ids, test_data = coset.load_test()
max_len = 50
language = 'es'

kim = KimModel(data_function=data_function, decode_function=coset.decode_label, persist_function=coset.persist_solution,
               test_function=coset.load_test)
kim.run(metrics=[metrics.fbeta_score], maxlen=max_len,
        batch_size=32, strides=1, filters=150, language=language,
        trainable=True,
        epochs=12, padding='same', dilation_rate=4, pool_size=8)
print(kim.test_f1_macro())
