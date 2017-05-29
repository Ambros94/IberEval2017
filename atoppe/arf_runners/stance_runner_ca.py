from data_loaders import stance
from deep_models import metrics
from deep_models.kim import KimModel
from nlp_utils.tweets_preprocessor import kim_stance_ca


def run(cleaning_function):
    data_function = stance.load_stance_ca
    test_function = stance.load_test_ca
    max_len = 30
    language = 'ca'

    kim = KimModel(data_function=data_function, decode_function=stance.decode_stance,
                   persist_function=None, test_function=test_function)
    kim.run(metrics=[metrics.fbeta_score], maxlen=max_len,
            batch_size=32, language=language, epochs=2, clean_tweets=cleaning_function)
    return kim.test_f1_macro()


print(run(kim_stance_ca))
