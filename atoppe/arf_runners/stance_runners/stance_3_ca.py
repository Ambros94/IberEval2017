from data_loaders import stance
from deep_models.cnn_lstm import CnnLstmModel

ids, data = stance.load_test_ca()
max_len = 35
language = 'ca'

stance_model = CnnLstmModel(data_function=stance.load_stance_ca, decode_function=stance.decode_stance,
                            persist_function=None, test_function=stance.load_test_ca)
stance_model.run(metrics=['categorical_accuracy'], maxlen=max_len,
                 embedding_size=128, kernel_size=5, dropout=0.25, strides=1, language=language,
                 filters=64, pool_size=4, lstm_output_size=70, batch_size=30, epochs=1)
stance_predictions = stance_model.predict(data=stance_model.x_persist, batch_size=32)
stance_decoded_predictions = [stance.decode_stance(p) for p in stance_predictions]

gender_model = CnnLstmModel(data_function=stance.load_gender_ca, decode_function=stance.decode_gender,
                            persist_function=None, test_function=stance.load_test_ca)
gender_model.run(metrics=['categorical_accuracy'], maxlen=max_len,
                 embedding_size=128, kernel_size=5, dropout=0.25, strides=1, language=language,
                 filters=64, pool_size=4, lstm_output_size=70, batch_size=30, epochs=1)
gender_predictions = gender_model.predict(data=gender_model.x_persist, batch_size=32)
g_decoded_predictions = [stance.decode_stance(p) for p in gender_predictions]

with open("stancecat17.atoppe.3.ca", 'w') as out_file:
    for id, stance, gender in zip(ids, g_decoded_predictions, stance_decoded_predictions):
        out_file.write("{id}:::{stance}:::{gender}\n".format(id=id, stance=stance, gender=gender))
