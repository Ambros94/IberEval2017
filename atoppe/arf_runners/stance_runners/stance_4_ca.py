from data_loaders import stance
from deep_models.fasttext import FastTextModel

ids, data = stance.load_test_ca()
max_len = 35
language = 'ca'

stance_model = FastTextModel(data_function=stance.load_stance_ca, decode_function=stance.decode_stance,
                             persist_function=None, test_function=stance.load_test_ca)
stance_model.run(metrics=['categorical_accuracy'], maxlen=max_len,
                 ngram_range=2, embedding_dims=300, hidden_dims=100, language=language,
                 batch_size=32, epochs=2)
stance_predictions = stance_model.predict(data=stance_model.x_persist, batch_size=32)
stance_decoded_predictions = [stance.decode_stance(p) for p in stance_predictions]

gender_model = FastTextModel(data_function=stance.load_gender_ca, decode_function=stance.decode_gender,
                             persist_function=None, test_function=stance.load_test_ca)
gender_model.run(metrics=['categorical_accuracy'], maxlen=max_len,
                              ngram_range=2, embedding_dims=300, hidden_dims=100, language=language,
                              batch_size=32, epochs=2)
gender_predictions = gender_model.predict(data=gender_model.x_persist, batch_size=32)
g_decoded_predictions = [stance.decode_stance(p) for p in gender_predictions]

with open("stancecat17.atoppe.4.ca", 'w') as out_file:
    for id, stance, gender in zip(ids, g_decoded_predictions, stance_decoded_predictions):
        out_file.write("{id}:::{stance}:::{gender}\n".format(id=id, stance=stance, gender=gender))
