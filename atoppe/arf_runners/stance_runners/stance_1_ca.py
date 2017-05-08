from data_loaders import stance
from deep_models import metrics
from deep_models.cnn import CNNModel

# Catalan stance
ids, data = stance.load_test_ca()
max_len = 35
language = 'ca'

stance_model = CNNModel(data_function=stance.load_stance_ca, decode_function=stance.decode_stance,
                        persist_function=None, test_function=stance.load_test_ca)
stance_model.run(metrics=[metrics.fbeta_score], maxlen=max_len,
                 batch_size=32, strides=1,
                 embedding_dims=50, filters=100,
                 kernel_size=3, dropout=0.2,
                 dropout_final=0.2, dilation_rate=1, padding='same',
                 hidden_dims=50, epochs=2, language=language)
stance_predictions = stance_model.predict(data=stance_model.x_persist, batch_size=32)
stance_decoded_predictions = [stance.decode_stance(p) for p in stance_predictions]

# Catalan gender

gender_model = CNNModel(data_function=stance.load_gender_ca, decode_function=stance.decode_gender,
                        persist_function=None, test_function=stance.load_test_ca)
gender_model.run(metrics=[metrics.fbeta_score], maxlen=max_len,
                 batch_size=32, strides=1,
                 embedding_dims=50, filters=100,
                 kernel_size=3, dropout=0.2,
                 dropout_final=0.2, dilation_rate=1, padding='same',
                 hidden_dims=50, epochs=2, language=language)
gender_predictions = gender_model.predict(data=gender_model.x_persist, batch_size=32)
g_decoded_predictions = [stance.decode_stance(p) for p in gender_predictions]

with open("stancecat17.atoppe.1.ca", 'w') as out_file:
    for id, stance, gender in zip(ids, g_decoded_predictions, stance_decoded_predictions):
        out_file.write("{id}:::{stance}:::{gender}\n".format(id=id, stance=stance, gender=gender))
