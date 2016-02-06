import numpy as np
from neuron import Neuron


class Network:
    nodes = []
    def __init__(self, layerSize, wList):

        for x in range(layerSize):
            self.nodes.append(Neuron(wList)) # create a node layer

        return

    def train(self, train_data, train_classes):
        """

        :param train_data:
        :param train_classes:
        :return: a list of 1s and 0s from the nodes for now
        """
        result = []

        for i in range(0, len(self.nodes)):
            # take a single instance of the data for every node
            result.append(self.nodes[i].calcOutput(train_data[i]))

        return result

    def predict(self, instances):
        return