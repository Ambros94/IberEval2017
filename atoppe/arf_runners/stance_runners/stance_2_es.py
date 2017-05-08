from data_loaders import stance
from deep_models import metrics

# Catalan stance
from deep_models.lstm import LSTMModel

ids, data = stance.load_test_es()
max_len = 35
language = 'es'

stance_model = LSTMModel(data_function=stance.load_stance_es, decode_function=stance.decode_stance,
                         persist_function=None, test_function=stance.load_test_es)
stance_model.run(metrics=[metrics.fbeta_score], maxlen=max_len, embedding_dims=100, language=language,
                 batch_size=32,
                 dropout=0.2, recurrent_dropout=0.4, lstm_units=128, epochs=2)
stance_predictions = stance_model.predict(data=stance_model.x_persist, batch_size=32)
stance_decoded_predictions = [stance.decode_stance(p) for p in stance_predictions]

# Catalan gender

gender_model = LSTMModel(data_function=stance.load_gender_es, decode_function=stance.decode_gender,
                         persist_function=None, test_function=stance.load_test_es)
gender_model.run(metrics=[metrics.fbeta_score], maxlen=max_len, embedding_dims=100, language=language,
                 batch_size=32,
                 dropout=0.2, recurrent_dropout=0.4, lstm_units=128, epochs=2)
gender_predictions = gender_model.predict(data=gender_model.x_persist, batch_size=32)
g_decoded_predictions = [stance.decode_stance(p) for p in gender_predictions]

with open("stancecat17.atoppe.1.es", 'w') as out_file:
    for id, stance, gender in zip(ids, g_decoded_predictions, stance_decoded_predictions):
        out_file.write("{id}:::{stance}:::{gender}\n".format(id=id, stance=stance, gender=gender))
