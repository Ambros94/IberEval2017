import csv

import numpy

from arf_runners.gender_runner_es import run as run_gender_es
from nlp_utils.tweets_preprocessor import *

K_folds = 3
pre_processing_functions = [nothing, sub_smiley, stemming, sub_hashtags, sub_numbers, stop_words, remove_url,
                            removed_reserved_words, sub_mentions, sub_emoji]
with open("../gender_ca.csv", 'a') as outcsv:
    csv.writer(outcsv, delimiter=';').writerow(["model_name", "cleaning_function", "mean_accuracy", "std_accuracy"])

for cleaning_function in pre_processing_functions:
    lstm = numpy.array([])
    cnn = numpy.array([])
    b_lstm = numpy.array([])
    cnn_lstm = numpy.array([])
    fast_text = numpy.array([])
    kim = numpy.array([])
    for i in range(K_folds):
        results = run_gender_es(cleaning_function)
        lstm = numpy.append(lstm, results[0])
        cnn = numpy.append(cnn, results[1])
        b_lstm = numpy.append(b_lstm, results[2])
        cnn_lstm = numpy.append(cnn_lstm, results[3])
        fast_text = numpy.append(fast_text, results[4])
        kim = numpy.append(kim, results[5])
    with open("../gender_es.csv", 'a') as outcsv:
        writer = csv.writer(outcsv, delimiter=';')
        writer.writerow(["lstm", cleaning_function.__name__, numpy.mean(lstm), numpy.std(lstm)])
        writer.writerow(["cnn", cleaning_function.__name__, numpy.mean(cnn), numpy.std(cnn)])
        writer.writerow(["b_lstm", cleaning_function.__name__, numpy.mean(b_lstm), numpy.std(b_lstm)])
        writer.writerow(["cnn_lstm", cleaning_function.__name__, numpy.mean(cnn_lstm), numpy.std(cnn_lstm)])
        writer.writerow(["fast_text", cleaning_function.__name__, numpy.mean(fast_text), numpy.std(fast_text)])
        writer.writerow(["kim", cleaning_function.__name__, numpy.mean(kim), numpy.std(kim)])
