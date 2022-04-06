from HB_perceptron import Perceptatron as Perceptron
import numpy as np
import os


class Main():

    name = ""
    perceptron = None
    labels = []
    input_data_labels = []
    input_data = None

    def __init__(self) -> None:
        self.show_welcome_screen()

    def display_data(self) -> None:
        print(f"Attribute labels: {self.labels}")
        for index in range(len(self.input_data)):
            print(
                f"Data set: {self.input_data_labels[index]} - Attributes: {self.input_data[index][:-1]} - Expected: {self.input_data[index][-1:]}")

    def show_welcome_screen(self) -> None:
        os.system("cls")
        print("===================================")
        print("Welcome to the dreamteam perceptron")
        print("===================================")
        print()
        self.automatic_training = input(
            "Would you like to use automatic training? (Y/N): ") == "y"

        if isinstance(self.perceptron, type(None)):
            if input("Do you want to use dummy data? (Y/N): ").lower() == "y":
                self.create_dummy_model()
            else:
                self.create_model()
        print()
        print(
            f"The model {self.name} has been created with the following data: ")
        self.display_data()
        print()
        self.perceptron = Perceptron(len(self.labels))

        user_command = input(
            "Enter Y to start training, or any other key to end programm: ").lower()
        while user_command == "y":            
            print(
                f"Prediction before learning - {self.perceptron.predict(self.input_data[:, :-1])}")
            self.perceptron.fit(
                self.input_data[:, :-1], self.input_data[:, -1:], self.input_data_labels, self.automatic_training)
            print(
                f"Prediction after learning - {self.perceptron.predict(self.input_data[:, :-1])}")
            user_command = input(
                "Enter Y to start a new training, or any other key to end programm: ").lower()
            print()

    def create_dummy_model(self):
        self.name = "Dragonfly flight response simulator"
        self.labels = ["Has 4 legs", "Is green", "Has a heart",
                       "Has a tail", "Is taller than 30cm", "Breathes fire"]
        self.input_data_labels = ["Cat", "Frog",
                                  "Andreas", "Dragon", "Flower", "Chair"]
        self.input_data = np.array([[1, 0, 1, 1, 1, 0, 1],
                                    [1, 1, 1, 0, 0, 0, 1],
                                    [0, 0, 1, 0, 1, 1, 0],
                                    [1, 0, 1, 1, 1, 1, 1],
                                    [0, 1, 0, 0, 0, 0, 0],
                                    [1, 1, 0, 0, 1, 0, 0]])

    def create_model(self) -> None:
        self.name = input(
            "What would you like to call your perceptron model: ")
        print(
            "To setup the model, you will be asked you what kind of attributes each data set should have")
        user_input = ""
        while user_input.lower() != "f":
            user_input = input(
                "Please enter an attribute name or an 'f' if you are finished entering attributes: ")
            if user_input.lower() != "f":
                self.labels.append(user_input)
        self.input_data = np.empty((0, len(self.labels) + 1))
        self.add_data_to_model()

    def add_data_to_model(self) -> None:
        print(
            "Lets add a data set")
        user_input = ""

        while user_input.lower() != "f":
            temp_array = np.zeros(len(self.labels)+1)
            self.input_data_labels.append(
                input("Please enter a name for the dataset: "))
            for index in range(len(self.labels)):
                temp_input = input(
                    f"Please enter a 0(=False) or 1(=True) for the attribute {self.labels[index]}: ")
                while int(temp_input) != 0 and int(temp_input) != 1:
                    temp_input = input(
                        f"Please enter a 0(=False) or 1(=True) for the attribute {self.labels[index]}: ")
                temp_array[index] = temp_input
            temp_array[len(self.labels)] = input(
                f"Please enter 0 if this dataset should give a negative resultat or 1 if it should be positive: ")
            while int(temp_array[len(self.labels)]) != 0 and int(temp_array[len(self.labels)]) != 1:
                temp_array[len(self.labels)] = input(
                    f"Please enter 0 if this dataset should give a negative resultat or 1 if it should be positive: ")
            self.input_data = np.append(self.input_data, [temp_array], axis=0)
            print(self.input_data)
            user_input = input(
                "Type 'f' to finish adding data or enter anything else to continue adding: ")

        self.display_data()


Main()
