import numpy as np
from neuron import Neuron
import random

class Network_Layer:
    nodes = []
    def __init__(self, layerSize, wSize):
        self.nodes = [] # creates a separate list for each class that is not shared
        for x in range(0, layerSize):
            self.nodes.append( Neuron( self.getRandList( wSize ) ) ) # create a node layer
        return

    def getRandList(self, numItems):
        weightList = np.zeros(numItems) # one weight for every value plus bias list[0]
        # get a small random number for the weights
        for i in range(0, len(weightList)):
            SRN = random.uniform(-1, 1)
            weightList[i] = SRN
        return weightList

    def train(self, train_data, train_classes):
        """

        :param train_data:
        :param train_classes:
        :return: a list of 1s and 0s
        """

        return

    def getOutputs(self, instance): # takes a single row and returns the results
     result = []
     for i in range(0, len(self.nodes)):
        # take a single instance of the data for every node
         result.append(self.nodes[i].calcOutput(instance))

     #print("Results for next node:", result)
     return result
