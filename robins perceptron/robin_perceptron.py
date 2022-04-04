import random
import numpy as np


class Robin_Perceptron:

    def __init__(self, number_of_inputs, training_session_epochs=30, learning_rate=0.15) -> None:
        self.weights = np.random.rand(number_of_inputs + 1)
        self.weights = np.around(self.weights, 2)
        # print(self.weights)
        self.training_session_epochs = training_session_epochs
        self.learning_rate = learning_rate

    def predict(self, inputs) -> int:
        sum_of_inputs_times_weights = np.dot(
            inputs, self.weights[1:]) + self.weights[0]
        if sum_of_inputs_times_weights > 0:
            return 1  # Activate!
        else:
            return 0  # Do not activate..

    def train(self, training_inputs: np.array, labels: np.array) -> None:
        for _ in range(self.training_session_epochs):
            for input, label in zip(training_inputs, labels):
                prediction = self.predict(input)
                self.weights[1:] += self.learning_rate * \
                    (label[0] - prediction) * input
                self.weights[0] += self.learning_rate * (label - prediction)
            print(self.weights)
