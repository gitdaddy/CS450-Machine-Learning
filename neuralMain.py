from sklearn import datasets
from Network import Network
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
    :return:
    """
    choice = int(input("1. Iris Data\n 2. Pima Data \n >"))

    # default values
    dataNormalized = []
    Targets = []

    if choice == 1:
        iris = datasets.load_iris()
        data = iris.data
        Targets = iris.target
        dataNormalized = nomalizer(data)
    else:
        # load pima set
        pData = readPimaData("pima.csv")
        Targets = pData[:, len(pData[0]) - 1]  # the last col
        dataNormalized = nomalizer(pData[:, :-1]) # everything but the last col

    #print("Pdata Normalized\n", dataNormalized, "Targets\n", Targets)


    weightList = np.zeros(len(dataNormalized[0]) + 1) # one weight for every value plus bias list[0]
    # get a small random number for the weights
    for i in range(0, len(weightList)):
        SRN = random.uniform(-1, 1)
        weightList[i] = SRN

    #print("Weights:", weightList)

    # number of nodes must be < num instances
    neural_classfier = Network(10, weightList)

    resultList = neural_classfier.train(dataNormalized, Targets)

    print("results:\n", resultList)

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