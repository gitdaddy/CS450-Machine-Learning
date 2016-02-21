import numpy as np
import math

class Neuron:
    def __init__(self, weights, isOutput = False):
        self.lastOutput = 0
        self.outputLayer = isOutput
        self.weights = weights
        self.newWeights = np.zeros(len(weights)) # weights to added

        self.error = 0
        return

    def sigmoid(self, h):
        #print("H:", h)
        if (abs(h) < 600):
            return 1 / (1 + math.pow(math.e, -h)) # 1 / (1 + e^(-x))
        else:
            print("H:", h)
            return 0.999999991

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

        self.lastOutput = self.sigmoid(sum)

        # if (sum > 4):
        #     print("Weights:", self.weights, "Instance:", instance)
        #     print("Sum:", sum, "output:", self.lastOutput)
        return self.lastOutput