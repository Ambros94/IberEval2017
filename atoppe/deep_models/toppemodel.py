import abc

from keras.models import load_model
from sklearn.metrics import f1_score, accuracy_score


class ToppeModel:
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_function, persist_function, decode_function, verbose=2):
        (self.ids_train, self.x_train, self.y_train), (
            self.ids_test, self.x_test, self.y_test) = data_function()
        if len(self.y_train) == 0:
            raise Exception("You should provide at least one train label")
        self.output_size = len(self.y_train[0])
        self.keras_model = None
        self.verbose = verbose
        self.persist_function = persist_function
        self.decode_function = decode_function

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

    def test_f1_macro(self):
        predictions = self.predict(data=self.x_test, batch_size=32)
        decoded_predictions = [self.decode_function(p) for p in predictions]
        decoded_ground_truth = [self.decode_function(p) for p in self.y_test]
        return f1_score(decoded_predictions, decoded_ground_truth, average='macro')

    def test_accuracy(self):
        predictions = self.predict(data=self.x_test, batch_size=32)
        decoded_predictions = [self.decode_function(p) for p in predictions]
        decoded_ground_truth = [self.decode_function(p) for p in self.y_test]
        correct, wrong = 0, 0
        # for i, p in enumerate(predictions):
        #    if decoded_predictions[i] != decoded_ground_truth[i]:
        #        print("Prediction:", predictions[i])
        #        print("Decoded prediction:", decoded_predictions[i])
        #        print("Ground truth:", self.y_test[i])
        #        print("Decoded Ground truth:", decoded_ground_truth[i])
        #        wrong += 1
        #    else:
        #        correct += 1
        # print(correct, wrong)

        return accuracy_score(decoded_ground_truth, decoded_predictions)

    def persist_result(self):
        predictions = self.predict(data=self.x_test, batch_size=32)
        self.persist_function(self.ids_test, predictions)
