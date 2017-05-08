from data_loaders import stance
from deep_models import metrics
from deep_models.cnn import CNNModel

data_function = stance.load_stance_ca
test_ids, test_data = stance.load_test_gender_ca()
max_len = 30
language = 'ca'

cnn = CNNModel(data_function=data_function, decode_function=stance.decode_stance,
               persist_function=stance.persist_stance)
cnn.run(metrics=[metrics.fbeta_score], maxlen=max_len,
        batch_size=32, strides=1,
        embedding_dims=50, filters=100,
        kernel_size=3, dropout=0.2,
        dropout_final=0.2, dilation_rate=1, padding='same',
        hidden_dims=50, epochs=2, language=language)
