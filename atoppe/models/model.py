import abc

from keras.models import load_model

import data_loaders.coset as c


class Model:
    __metaclass__ = abc.ABCMeta

    def __init__(self, data, verbose=2):
        (self.ids_train, self.x_train, self.y_train), (self.ids_val, self.x_val, self.y_val), (
            self.ids_test, self.x_test, self.y_test) = data
        if len(self.y_train) == 0:
            raise Exception("You should provide at least one train label")
        self.output_size = len(self.y_train[0])
        self.model = None
        self.verbose = verbose

    def run(self, batch_size, epochs, **params):
        self.build(params)
        self.train(batch_size=batch_size, epochs=epochs)
        return self.evaluate_val(batch_size=batch_size)

    @abc.abstractmethod
    def build(self, params):
        """Build a model providing all necessary parameters"""
        raise Exception("This is an abstract method!")

    def load_model(self, name):
        self.model = load_model(name)

    def persist_model(self, name):
        self.model.save(name)

    def train(self, batch_size, epochs):
        if self.model is None:
            raise Exception("Cannot find a model! Have you build it yet?")
        self.model.fit(self.x_train, self.y_train,
                       verbose=self.verbose,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(self.x_val, self.y_val))

    def evaluate_val(self, batch_size):
        if self.model is None:
            raise Exception("Cannot find a model! Have you build it yet?")
        return self.model.evaluate(self.x_val, self.y_val,
                                   batch_size=batch_size)

    def evaluate_test(self, batch_size):
        if self.model is None:
            raise Exception("Cannot find a model! Have you build it yet?")
        return self.model.evaluate(self.x_test, self.y_test,
                                   batch_size=batch_size)

    def predict(self, data, batch_size):
        if self.model is None:
            raise Exception("Cannot find a model! Have you build it yet?")
        return self.model.predict(data,
                                  batch_size=batch_size)

    def persist_result(self):
        c.persist_solution(self.ids_test, self.predict(data=self.x_test, batch_size=32))
