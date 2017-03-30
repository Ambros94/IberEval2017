import abc


class Model:
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_function):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data_function()
        if len(self.y_train) == 0:
            raise Exception("You should provide at least one train label")
        self.output_size = len(self.y_train[0])
        self.model = None

    def run(self, batch_size, epochs, **params):
        self.build(params)

        self.train(batch_size=batch_size, epochs=epochs)
        return self.evaluate(batch_size=batch_size)

    @abc.abstractmethod
    def build(self, params):
        """Build a model providing all necessary parameters"""
        return

    def train(self, batch_size, epochs):
        if self.model is None:
            raise Exception("Cannot find a model! Have you build it yet?")
        self.model.fit(self.x_train, self.y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(self.x_test, self.y_test))

    def evaluate(self, batch_size):
        return self.model.evaluate(self.x_test, self.y_test,
                                   batch_size=batch_size)
