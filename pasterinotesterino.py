"""
MIT License
Copyright (c) 2018 Thomas Countz
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import unittest
import numpy as np
from copypasterino import Perceptron


class PerceptronTest(unittest.TestCase):

    def test_mimics_logical_and(self):
        weights = np.array([-1, 1, 1])

        a = 1
        b = 1
        inputs = np.array([a, b])

        perceptron = Perceptron(inputs.size)
        perceptron.weights = weights

        output = perceptron.predict(inputs)
        self.assertEqual(output, a & b)

    def test_trains_for_logical_and(self):
        labels = np.array([1, 0, 0, 0])
        input_matrix = []
        input_matrix.append(np.array([1, 1]))
        input_matrix.append(np.array([1, 0]))
        input_matrix.append(np.array([0, 1]))
        input_matrix.append(np.array([0, 0]))

        perceptron = Perceptron(2, threshold=10, learning_rate=1)
        perceptron.train(input_matrix, labels)

        a = 1
        b = 1
        inputs = np.array([a, b])

        output = perceptron.predict(inputs)
        self.assertEqual(output, a & b)


if __name__ == '__main__':
    unittest.main()
