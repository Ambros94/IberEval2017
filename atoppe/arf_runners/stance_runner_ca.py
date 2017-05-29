from data_loaders import stance
from deep_models import metrics
from deep_models.kim import KimModel
from nlp_utils.tweets_preprocessor import kim_stance_es


def run(cleaning_function):
    data_function = stance.load_stance_ca
    test_function = stance.load_test_ca
    max_len = 20
    language = 'ca'

    kim = KimModel(data_function=data_function, decode_function=stance.decode_stance,
                   persist_function=None, test_function=test_function)
    kim.run(metrics=[metrics.fbeta_score], maxlen=max_len,
            batch_size=32, filters=150, language=language, epochs=3, padding='same', dilation_rate=1, pool_size=5,
            clean_tweets=cleaning_function)
    kim_accuracy = kim.test_f1_macro()
    return kim_accuracy


print(run(kim_stance_es))
