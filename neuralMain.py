from sklearn import datasets
from Network import Network_Layer
import numpy as np
import csv, sys
import random

def readPimaData(fileName):
    with open(fileName, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            temp = []
            for row in spamreader:
                for x in row:
                    # TODO handle missing data
                    temp.append(float(x))

    return np.reshape(temp, (-1, 9)) # turn a 1 D array to a 2D matrix


def testNeuralNetwork():
    """
    This function will load the dataSets and test them
    Build Layers, first layer # nodes is determined by the # attributes
    list of weights for each instance plus the bias input
    :return:
    """
    choice = int(input("1. Iris Data\n 2. Pima Data \n >"))
    # default values
    dataNormalized = []
    Targets = []
    layers = [] # list of network layers
    predictions = []
    numCorrect = 0

    if choice == 1:
        iris = datasets.load_iris()
        data = iris.data
        Targets = iris.target
        dataNormalized = nomalizer(data)

        layers.append(Network_Layer(len(dataNormalized[0]), len(dataNormalized[0]) + 1))
        # second layer
        layers.append(Network_Layer(3, len(layers[0].nodes) + 1))

        for i in range(0, len(dataNormalized)):
            #print("Current Instance:", dataNormalized[i])
            predictions.append(np.argmax(layers[1].getOutputs(layers[0].getOutputs(dataNormalized[i]))))
            if (predictions[i] == Targets[i]):
                numCorrect += 1
    else:
         # load pima set
         pData = readPimaData("pima.csv")
         Targets = pData[:, len(pData[0]) - 1]  # the last col
         dataNormalized = nomalizer(pData[:, :-1]) # everything but the last col
         layers.append(Network_Layer(len(dataNormalized[0]), len(dataNormalized[0]) + 1))
         # second layer
         layers.append(Network_Layer(3, len(layers[0].nodes) + 1))
         # Third layer output layer
         layers.append(Network_Layer(1, len(layers[1].nodes) + 1))

         for i in range(0, len(dataNormalized)):
             #print("Current Instance:", dataNormalized[i])
             predictions.append(np.argmax(layers[2].getOutputs(layers[1].getOutputs(layers[0].getOutputs(dataNormalized[i])))))
             if (predictions[i] == Targets[i]):
                 numCorrect += 1


    #print("Pdata Normalized\n", dataNormalized, "Targets\n", Targets)

    print("Percent Correct: %.2f" %(numCorrect/len(Targets)), " Number Correct:", numCorrect)


    return



def nomalizer(dataSet):
    """
    TODO: This function will normalize the data
    the data should be in the form of a matrix
    with numeric data
    :param dataSet:
    :return:
    """
    numRow = len(dataSet)
    numCol = len(dataSet[0])
    temp_z = np.zeros((numRow, numCol))
    for col in range(0, numCol):
            x_np = np.asarray(dataSet[:,col])
            tempZCol = (x_np - x_np.mean()) / x_np.std()
            temp_z[:,col] = tempZCol # assign one column at a time

    return temp_z

def main(argv):
    """
    Neural Networks Preceptron Algorithm
    Data should be in a numeric form

    :param argv:
    :return:
    """
    testNeuralNetwork()
    return


if __name__ == "__main__":
    main(sys.argv)