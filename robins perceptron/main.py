from robin_perceptron import Robin_Perceptron as Perceptron
from input_node import Input_Node
import numpy as np


class Main():

    data_available = False
    name = ""
    perceptron = None
    labels = []
    input_data = None

    def __init__(self) -> None:
        self.show_welcome_screen()

    def load_data(self) -> None:
        print("DATA")

    def show_welcome_screen(self) -> None:
        print("Welcome to robins perceptron")
        if isinstance(self.perceptron, type(None)):
            self.create_model()
        self.perceptron = Perceptron(len(self.labels))
        self.perceptron.train(self.input_data[:, :-1], self.input_data[:, -1:])

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
        # self.input_data = np.array([[1, 1, 0, 1, 1], [0, 0, 1, 1, 1], [1, 0, 0, 1, 0], [
        #                            1, 0, 1, 1, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        user_input = ""

        while user_input.lower() != "f":
            temp_array = np.zeros(len(self.labels)+1)
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

    def save_new_data() -> None:
        input_nodes = []

        # self.perceptron = Perceptron(len(input_nodes))


test = Main()
