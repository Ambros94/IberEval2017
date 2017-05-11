from keras.layers import Dense, Embedding, GlobalAveragePooling1D
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from data_loaders import coset
from deep_models import metrics
from nlp_utils import word_vectors
from nlp_utils.n_grams import augment_with_n_grams
from nlp_utils.tweets_preprocessor import clean_tweets


def create_model(neurons=1):
    model = Sequential()
    model.add(Embedding(max_features, 300, trainable=True, weights=[embedding_matrix],
                        input_length=30))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[metrics.fbeta_score])
    return model


# Load and prepare data
ids, x, y = coset.load_data()
x = clean_tweets(x)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)
x = tokenizer.texts_to_sequences(x)
x, max_features = augment_with_n_grams(x, max_features=7686, ngram_range=2)
x = sequence.pad_sequences(x, maxlen=30)

embedding_matrix = word_vectors.load_vectors(tokenizer.word_index, language='es', num_words=max_features)

# Build the model
classifier = KerasClassifier(build_fn=create_model, epochs=2, batch_size=32, verbose=0)

# define the grid search parameters
neurons = [64, 128, 256]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=4)
grid_result = grid.fit(x, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
