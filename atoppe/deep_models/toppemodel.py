import abc

from keras.models import load_model
from sklearn.metrics import f1_score

import data_loaders.coset as c


class ToppeModel:
    __metaclass__ = abc.ABCMeta

    def __init__(self, data, verbose=2):
        (self.ids_train, self.x_train, self.y_train), (
            self.ids_test, self.x_test, self.y_test) = data
        if len(self.y_train) == 0:
            raise Exception("You should provide at least one train label")
        self.output_size = len(self.y_train[0])
        self.keras_model = None
        self.verbose = verbose

    @abc.abstractmethod
    def build(self, params):
        """Build a model providing all necessary parameters"""
        raise Exception("This is an abstract method!")

    def run(self, batch_size, epochs, **params):
        self.build(params)
        self.train(batch_size=batch_size, epochs=epochs)

    def load_model(self, name):
        self.keras_model = load_model(name)

    def persist_model(self, name):
        self.keras_model.save(name)

    def train(self, batch_size, epochs):
        if self.keras_model is None:
            raise Exception("Cannot find a model! Have you build it yet?")
        self.keras_model.fit(self.x_train, self.y_train,
                             verbose=self.verbose,
                             batch_size=batch_size,
                             epochs=epochs,
                             validation_data=(self.x_test, self.y_test))

    def evaluate_test(self, batch_size):
        if self.keras_model is None:
            raise Exception("Cannot find a model! Have you build it yet?")
        return self.keras_model.evaluate(self.x_test, self.y_test,
                                         batch_size=batch_size)

    def predict(self, data, batch_size):
        if self.keras_model is None:
            raise Exception("Cannot find a model! Have you build it yet?")
        return self.keras_model.predict(data,
                                        batch_size=batch_size)

    def test_f1_micro(self):
        predictions = self.predict(data=self.x_test, batch_size=32)
        sk_f1_micro = f1_score(c.decode_labels(self.y_test), c.decode_labels(predictions), average='micro')
        return sk_f1_micro

    def test_f1_macro(self):
        predictions = self.predict(data=self.x_test, batch_size=32)
        return f1_score(c.decode_labels(self.y_test), c.decode_labels(predictions), average='macro')

    def persist_result(self):
        predictions = self.predict(data=self.x_test, batch_size=32)
        score = f1_score(c.decode_labels(self.y_test), c.decode_labels(predictions), average='macro')
        print('* The macro F1-score achieved is: {:.4f}'.format(score))
        c.persist_solution(self.ids_test, predictions)
