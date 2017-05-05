from data_loaders import coset
from models.cnnv2 import KimModel

max_words = 9000
max_len = 30

data = coset.load_data(pre_process=True, use_nltk=True)

cnn2 = KimModel(data=data)
cnn2_f1_micro = cnn2.run(metrics=[coset.fbeta_score], max_words=max_words, maxlen=max_len,
                         batch_size=32, strides=1, embedding_dims=150, filters=150,
                         dropout=0.5, dropout_final=0.5, trainable=True,
                         recurrent_units=128, epochs=3, padding='same', dilation_rate=3, pool_size=5)

print(cnn2_f1_micro)
