import numpy

from data_loaders import coset
from deep_models import metrics
from deep_models.fasttext import FastTextModel

data_function = coset.load_data
test_ids, test_data = coset.load_test()
language = 'es'


def run_model(max_len_array, epochs_array, hidden_dims_array, runs_count):
    runs = numpy.array([])
    stats = []
    for i in range(runs_count):
        for max_len in max_len_array:
            for epochs in epochs_array:
                for hidden_dims in hidden_dims_array:
                    fast_text = FastTextModel(data_function=data_function, decode_function=coset.decode_label,
                                              persist_function=coset.persist_solution,
                                              test_function=coset.load_test)
                    fast_text.run(metrics=[metrics.fbeta_score], maxlen=max_len,
                                  ngram_range=2, embedding_dims=300, hidden_dims=hidden_dims, language=language,
                                  batch_size=32, epochs=epochs)
                    runs = numpy.append(runs, fast_text.test_f1_macro())
                    res = "Avg:{avg}, Std:{std} , Max:{max}, max_len:{max_len}, epochs:{epochs}, hidden_dims:{hidden_dims}".format(
                        avg=numpy.mean(runs), std=numpy.std(runs),
                        max=numpy.max(runs), max_len=max_len, epochs=epochs, hidden_dims=hidden_dims)
                    print(res)
                    stats.append(res)
    return stats


stat = run_model(max_len_array=[25, 30, 35, 40, 45, 50], epochs_array=[7, 8, 9], hidden_dims_array=[256], runs_count=3)
for run in stat:
    print(run)
