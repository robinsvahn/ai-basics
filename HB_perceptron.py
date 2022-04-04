import numpy as np

class Perceptatron:
    def __init__(self, n_features, learning_rate=0.01, n_iterations=100):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = np.random.ranf(n_features)

    def predict(self, inputs):
        linear_output = np.dot(inputs, self.weights)                   # sum of inputs and their weights
        predicted = _unit_step_func(linear_output)                     # 1 if linear_output exceeds threshold, else 0
        return predicted

    def fit(self, inputs, labels, automatic=True):
        for _ in range(self.n_iterations):
            for i, input_i in enumerate(inputs):
                predicted = self.predict(input_i)
                if automatic:
                    update = self.lr * (labels[i] - predicted)        # Perceptron update rule when automatic
                else:
                    print("For: " + str(input_i) + " the Perceptotron predict the output to be " + str(predicted))
                    answer = int(input("What is the expected output? (0/1)"))
                    update = self.lr * (answer - predicted)           # Perceptron update rule when manual
                temp = np.array(update * input_i)
                self.weights += temp                                  # adjusting weights


def _unit_step_func(x, threshold=1):
    return np.where(x > threshold, 1, 0)                              # if condition met return 1, else 0

