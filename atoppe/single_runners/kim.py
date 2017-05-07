from data_loaders import coset
from deep_models import metrics
from deep_models.kim import KimModel

max_len = 30

data = coset.load_data()

kim = KimModel(data=data)
kim.run(metrics=[metrics.fbeta_score], maxlen=max_len,
        batch_size=32, strides=1, filters=150,
        dropout=0.5, dropout_final=0.5, trainable=True, language='es',
        recurrent_units=128, epochs=3, padding='same', dilation_rate=3, pool_size=5)
kim_f1_micro = kim.test_f1_macro()

print(kim_f1_micro)
