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
    choice = int(input("1. Iris Data\n2. Pima Data \n>"))
    # default values
    trainSet = {'train_data': [], 'train_classes': []}
    validSet = {'valid_data': [], 'valid_classes': []}
    testSet = {'test_data': [], 'test_classes': []}

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
    else:
         # load pima set
         pData = readPimaData("pima.csv")
         Targets = pData[:, len(pData[0]) - 1]  # the last col
         dataNormalized = nomalizer(pData[:, :-1]) # everything but the last col

    testNum = int(len(dataNormalized) * 0.3)  # 50% train, 20% validate, 30% test
    valNum = int(len(dataNormalized) * 0.2)
    trainNum = int(len(dataNormalized) * 0.5)

    #randomize the data
    indices = np.random.permutation(len(dataNormalized))

    for i, item in enumerate(indices):
        if (i < trainNum):
            trainSet['train_data'].append(dataNormalized[item])
            trainSet['train_classes'].append(Targets[item])
        elif (i < (trainNum + valNum)):
            validSet['valid_data'].append(dataNormalized[item])
            validSet['valid_classes'].append(Targets[item])
        else:
            testSet['test_data'].append(dataNormalized[item])
            testSet['test_classes'].append(Targets[item])

    # set the layers
    irisLayers = [len(dataNormalized[0]), 2, 3]
    pimaLayers = [len(dataNormalized[0]), 1]
    finalAcc = 0

    if (choice == 1):
        irisNetwork = Network_Layer(irisLayers)
        irisNetwork.train(trainSet['train_data'], trainSet['train_classes'], validSet['valid_data'], validSet['valid_classes'], True)
        finalAcc = irisNetwork.getAcc(testSet['test_data'], testSet['test_classes'], True)

    else:
        pimaNetwork = Network_Layer(pimaLayers)
        pimaNetwork.train(trainSet['train_data'], trainSet['train_classes'], validSet['valid_data'], validSet['valid_classes'], False)
        finalAcc = pimaNetwork.getAcc(testSet['test_data'], testSet['test_classes'], False)


    #print("Predictions:", predictions)
    print("Final Percent Correct: %.2f" %(finalAcc))

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
            x_np = np.asarray(dataSet[:, col])
            tempZCol = (x_np - x_np.mean()) / x_np.std()
            temp_z[:, col] = tempZCol # assign one column at a time

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