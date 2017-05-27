import csv
import inspect
from time import gmtime, strftime

from data_loaders import coset
from deep_models import metrics
from deep_models.fasttext import FastTextModel


def run(cleaning_function):
    data_function = coset.load_data
    max_len = 30
    language = 'ca'
    fast_text = FastTextModel(data_function=data_function, decode_function=coset.decode_label,
                              persist_function=coset.persist_solution,
                              test_function=coset.load_test)
    fast_text.run(metrics=[metrics.fbeta_score], maxlen=max_len,
                  ngram_range=2, hidden_dims=128, language=language, noise=0.2, clean_tweets=cleaning_function,
                  batch_size=32, epochs=8)
    fast_text_f1_macro = fast_text.test_f1_macro()

    # kim = KimModel(data_function=data_function, decode_function=coset.decode_label,
    #               persist_function=coset.persist_solution,
    #               test_function=coset.load_test)
    # kim.run(metrics=[metrics.fbeta_score], maxlen=max_len,
    #        batch_size=32, strides=1, filters=150, language=language,
    #        dropout=0.5, trainable=True, clean_tweets=cleaning_function,
    #        epochs=4, padding='same', dilation_rate=4, pool_size=8)
    # kim_f1_macro = kim.test_f1_macro()

    with open("../coset-" + strftime("%Y%m%d_%H%M%S", gmtime()) + ".log", 'w') as outcsv:
        writer = csv.writer(outcsv, delimiter=';')
        writer.writerow(["model_name", "preprocessing", "test_f1_macro"])
        # writer.writerow(["kim", cleaning_function.__name__, ('%.3f' % kim_f1_macro).replace('.', ',')])
        writer.writerow(["fasttext", cleaning_function.__name__, ('%.3f' % fast_text_f1_macro).replace('.', ',')])
        outcsv.write("Pre-processing:")
        outcsv.write(''.join(inspect.getsourcelines(cleaning_function)[0]))
