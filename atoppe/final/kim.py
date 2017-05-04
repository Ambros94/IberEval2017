from data_loaders import coset
from models.cnnv2 import CNNModelv2

max_features = 9000
max_len = 30

data = coset.load_data(pre_process=True, use_nltk=False)

cnn2 = CNNModelv2(data=data)
cnn2_f1_micro = cnn2.run(metrics=[coset.fbeta_score], max_features=max_features, maxlen=max_len,
                         batch_size=32, strides=1,
                         embedding_dims=150, filters=150,
                         dropout=0.2,
                         dropout_final=0.2,
                         hidden_dims=50, epochs=3, padding='same', dilation_rate=1, pool_size=3)

print(cnn2_f1_micro)
