import random

class Input_Node:
    
    def __init__(self, name: str) -> None:
        self.name = name
        self.weight = random.random()