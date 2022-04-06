import numpy as np


class Perceptatron:
    def __init__(self, n_features, learning_rate=0.01, n_iterations=10):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = np.random.ranf(n_features)

    def predict(self, inputs):
        # sum of inputs and their weights
        linear_output = np.dot(inputs, self.weights)
        # 1 if linear_output exceeds threshold, else 0
        predicted = _unit_step_func(linear_output)
        return predicted

    def fit(self, inputs, labels, dataset_labels, automatic=True):
        for _ in range(self.n_iterations):
            for i, input_i in enumerate(inputs):
                predicted = self.predict(input_i)
                if automatic:
                    update = self.lr * (labels[i] - predicted)
                else:
                    print("For: " + str(dataset_labels[i]) +
                          " the perceptron predict the output to be " + str(predicted))
                    answer = int(input("What is the expected output? (0/1)"))
                    # Perceptron update rule when manual
                    update = self.lr * (answer - predicted)
                temp = np.array(update * input_i)
                self.weights += temp                                  # adjusting weights


def _unit_step_func(x, threshold=1):
    # if condition met return 1, else 0
    return np.where(x > threshold, 1, 0)
