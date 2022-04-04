import numpy as np
from HB_perceptron import Perceptatron


def get_user_samples():
    n_samples = int(input("How many entries/samples: "))
    n_features = int(input("How many attributes: "))
    inputs = np.zeros(n_features * n_samples).reshape(n_samples, n_features)

    for n in range(n_samples):
        temp = np.zeros(n_features)                         # creating array with n zeros at the start of loop
        print("Enter attributes for entry " + str(n+1))
        for i in range(n_features):
            attr = input("Attribute " + str(i+1) + ": ")    # getting attribute input from user
            temp[i] = attr                                  # temporary storage of entered attributes
        inputs[n] = temp                                    # adding the entry to total list of samples
    return inputs


# MANUALLY TRAINING ##########################################################
# manual training of perceptron and user input sample
inputs = get_user_samples()                                 # getting the sample from the user
m_labels = None                                             # set to None for manual fitting
percipis_m = Perceptatron(inputs.shape[1], learning_rate=0.2, n_iterations=5)

print(percipis_m.predict(inputs))                           # first guess
percipis_m.fit(inputs, m_labels, automatic=False)           # manual fitting och perceptron
print(percipis_m.predict(inputs))                           # after fit guess


# AUTOMATIC TRAINING ##########################################################
# automatic training of perceptron and pre-given sample
data_set = np.array([[1, 0, 1],                             # sample set with 3 given attr per entry
                     [0, 1, 0],
                     [1, 1, 1],
                     [0, 0, 0],
                     [0, 1, 1],
                     [0, 1, 0],
                     [1, 1, 0],
                     [0, 0, 1],
                     [1, 1, 1],
                     [1, 0, 0],
                     [1, 1, 0],
                     [1, 0, 1]])

expected_predict = np.array([1, 0, 1, 0, 1, 0,              # array with the correct answers
                             1, 0, 1, 0, 1, 1])

percipis_a = Perceptatron(3)                                # creating perceptron that takes 3 attributes
print(percipis_a.predict(data_set))                         # first guess
percipis_a.fit(data_set, expected_predict)                  # automatic training
print(percipis_a.predict(data_set))                         # after fit guess
