import numpy as np
import math

class Neuron:
    weights = [] # list of weights for bias = w[0]
    def __init__(self, weights):
        self.weights = []
        self.weights = weights
        return

    def sigmoidifier(self, h):
        return 1 / (1 + math.pow(math.e, -h)) # 1 / (1 + e^(-x))

    def calcOutput(self, instance):
        """
        Takes in a single instance with arbitrary
        number of attributes and returns
        :param instance:
        :return:
        """
        #print("Num Weights:", len(self.weights), " Num cols:", len(instance))

        sum = -1 * self.weights[0] # bias here

        for i in range(0, len(instance)):
            sum += self.weights[i+1] * instance[i]

        #print("Sum = ", sum)
        return self.sigmoidifier(sum)