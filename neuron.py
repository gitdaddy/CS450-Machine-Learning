import numpy as np

class Neuron:
    biasInput = 0
    weights = [] # list of weights for bias = w[0]
    threshold = 0

    def __init__(self, weights, bias=1):
        self.biasInput = bias
        self.weights = weights
        threshold = 0
        return

    def calcOutput(self, instance):
        """
        Takes in a single instance with arbitrary
        number of attributes and returns 1 or 0 if
        threshold is meet
        :param instance:
        :return:
        """
        sum = self.biasInput * self.weights[0]
        for i in range(0, len(instance)):
            sum += self.weights[i+1] * instance[i]

        print("Sum = ", sum)
        return int(sum > self.threshold)