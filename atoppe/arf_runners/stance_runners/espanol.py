from data_loaders import stance
from deep_models import metrics
from deep_models.bidirectional_lstm import BidirectionalLSTMModel
from deep_models.cnn import CNNModel

# Catalan stance
from nlp_utils.tweets_preprocessor import cl, st_sw_cl

max_len = 30
language = 'es'

stance_model = BidirectionalLSTMModel(data_function=stance.load_stance_es, decode_function=stance.decode_stance,
                                      persist_function=None, test_function=stance.load_test_es)
stance_model.run(metrics=[metrics.fbeta_score], max_len=max_len,
                 batch_size=32, recurrent_units=64, dropout=0.1, language=language, clean_tweets=st_sw_cl,
                 epochs=2)
print("Stance f1-macro:", stance_model.test_f1_macro())
stance_predictions = stance_model.predict(data=stance_model.x_persist, batch_size=32)
stance_decoded_predictions = [stance.decode_stance(p) for p in stance_predictions]

# Catalan gender

gender_model = CNNModel(data_function=stance.load_gender_es, decode_function=stance.decode_gender,
                        persist_function=None, test_function=stance.load_test_es)
gender_model.run(metrics=['categorical_accuracy'], maxlen=max_len,
                 batch_size=32, strides=1, filters=100,
                 kernel_size=3, dropout=0.2,
                 dropout_final=0.2, dilation_rate=1, padding='same',
                 hidden_dims=50, epochs=2, language=language, clean_tweets=cl)
print("Gender accuracy:", gender_model.test_accuracy())
gender_predictions = gender_model.predict(data=gender_model.x_persist, batch_size=32)
g_decoded_predictions = [stance.decode_gender(p) for p in gender_predictions]
with open("stancecat17.atoppe.1.es", 'w') as out_file:
    for id, gender, stance in zip(gender_model.ids_persist, g_decoded_predictions, stance_decoded_predictions):
        out_file.write("{id}:::{stance}:::{gender}\n".format(id=id, stance=stance, gender=gender))
