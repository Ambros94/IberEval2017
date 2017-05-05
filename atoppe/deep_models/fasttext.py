from keras.layers import Embedding, Dense
from keras.layers import GlobalAveragePooling1D
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from deep_models.toppemodel import ToppeModel
from nlp_utils import word_vecors
from nlp_utils.n_grams import augment_with_n_grams
from nlp_utils.tweets_preprocessor import clean_tweets


class FastTextModel(ToppeModel):
    def build(self, params):
        # Extract params
        language = params['language']
        ngram_range = params['ngram_range']
        maxlen = params['maxlen']
        embedding_dims = params['embedding_dims']
        hidden_dims = params['hidden_dims']
        # Cleaning data
        self.x_train = clean_tweets(self.x_train)
        self.x_test = clean_tweets(self.x_test)
        # Prepare data
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.x_train)
        x_train = tokenizer.texts_to_sequences(self.x_train)
        x_test = tokenizer.texts_to_sequences(self.x_test)
        print('Found {word_index} unique tokens'.format(word_index=len(tokenizer.word_index)))
        self.x_train, self.x_test, max_features = augment_with_n_grams(x_train=x_train, x_test=x_test,
                                                                       max_features=len(tokenizer.word_index),
                                                                       ngram_range=ngram_range)
        print('After {n}_grams we have {max_features} features'.format(n=ngram_range, max_features=max_features))
        embedding_matrix = word_vecors.load_vectors(tokenizer.word_index, language=language, num_words=max_features)

        self.x_train = sequence.pad_sequences(self.x_train, maxlen=maxlen)
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=maxlen)
        # Model creation
        self.keras_model = Sequential()

        self.keras_model.add(Embedding(max_features,
                                       embedding_dims,
                                       weights=[embedding_matrix],
                                       input_length=maxlen))

        self.keras_model.add(GlobalAveragePooling1D())
        self.keras_model.add(Dense(hidden_dims, activation='relu'))
        self.keras_model.add(Dense(self.output_size, activation='softmax'))

        self.keras_model.compile(loss='categorical_crossentropy',
                                 optimizer='adam',
                                 metrics=params['metrics'])
