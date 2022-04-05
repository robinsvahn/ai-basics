import numpy as np
from HB_perceptron import Perceptatron


def get_user_samples():
    n_samples = int(input("How many entries/samples: "))
    n_features = int(input("How many attributes: "))
    inputs = np.zeros(n_features * n_samples).reshape(n_samples, n_features)

    for n in range(n_samples):
        # creating array with n zeros at the start of loop
        temp = np.zeros(n_features)
        print("Enter attributes for entry " + str(n+1))
        for i in range(n_features):
            # getting attribute input from user
            attr = input("Attribute " + str(i+1) + ": ")
            # temporary storage of entered attributes
            temp[i] = attr
        # adding the entry to total list of samples
        inputs[n] = temp
    return inputs


# MANUALLY TRAINING ##########################################################
# manual training of perceptron and user input sample
# getting the sample from the user
inputs = get_user_samples()
# set to None for manual fitting
m_labels = None
percipis_m = Perceptatron(inputs.shape[1], learning_rate=0.2, n_iterations=5)

print(percipis_m.predict(inputs))                           # first guess
# manual fitting by asking for correct label from user
percipis_m.fit(inputs, m_labels, automatic=False)
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

# creating perceptron that takes 3 attributes
percipis_a = Perceptatron(3)
print(percipis_a.predict(data_set))                         # first guess
# automatic training
percipis_a.fit(data_set, expected_predict)
print(percipis_a.predict(data_set))                         # after fit guess
