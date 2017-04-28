import abc

from keras.models import load_model

from atoppe.data_loaders import coset


class Model:
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_function, verbose=2):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data_function()
        if len(self.y_train) == 0:
            raise Exception("You should provide at least one train label")
        self.output_size = len(self.y_train[0])
        self.model = None
        self.verbose = verbose

    def run(self, batch_size, epochs, **params):
        self.build(params)
        self.train(batch_size=batch_size, epochs=epochs)
        return self.evaluate(batch_size=batch_size)

    @abc.abstractmethod
    def build(self, params):
        """Build a model providing all necessary parameters"""
        raise Exception("This is an abstract method!")

    def load_model(self, name):
        self.model = load_model(name)

    def persist_model(self, name):
        self.model.save(name)

    def evaluate_on_task(self, batch_size):
        ids, tweets=coset
        predictions = self.model.predict(test_set, batch_size=batch_size)

        # Load the ground truth
        true_ids, true_labels = coset.load_ground_truth()


    def train(self, batch_size, epochs):
        if self.model is None:
            raise Exception("Cannot find a model! Have you build it yet?")
        self.model.fit(self.x_train, self.y_train,
                       verbose=self.verbose,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(self.x_test, self.y_test))

    def evaluate(self, batch_size):
        if self.model is None:
            raise Exception("Cannot find a model! Have you build it yet?")
        return self.model.evaluate(self.x_test, self.y_test,
                                   batch_size=batch_size)
