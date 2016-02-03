from sklearn import datasets
import itertools, random
import numpy as np
from Classifier import Classifier
from TreeClassifier import TreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy import special, optimize
import pandas as pd
import csv, sys

def readCarData():
    with open('car.csv', 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            temp = []
            for row in spamreader:
                temp.append(row)
    return temp

def readLensData(fileName):
    with open(fileName, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            temp = []
            for row in spamreader:
                for x in row:
                    temp.append(int(x))

    return np.reshape(temp, (-1, 6)) # turn a 1 D array to a 2D matrix

def readVotesData(fileName):
    data = []
    with open(fileName, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                data.append(row)

    return np.reshape(data, (-1, 17)) # turn a 1 D array to a 2D matrix

def replace(l, X, Y): # for a 2D matrix
  for i in range(0, len(l)):
      for j in range(0, len(l[0])):
          if l[i][j] == X:
              l[i][j] = Y

def encodeCar(preCarData):
    # scale the data
    replace(preCarData, 'vhigh', 4)
    replace(preCarData, 'high', 3)
    replace(preCarData, 'med', 2)
    replace(preCarData, 'low', 1)
    replace(preCarData, 'small', 1)
    replace(preCarData, 'big', 3)
    replace(preCarData, '5more', 6)
    replace(preCarData, 'more', 5)
    replace(preCarData, '1', 1)
    replace(preCarData, '2', 2)
    replace(preCarData, '3', 3)
    replace(preCarData, '4', 4)
    replace(preCarData, '5', 5)
    replace(preCarData, 'acc', 2)
    replace(preCarData, 'unacc', 1)
    replace(preCarData, 'good', 3)
    replace(preCarData, 'vgood', 4)
    return

def writeCSV(data):
    with open('lenses.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for row in range(0, len(data)):
            spamwriter.writerow(data[row])

def testKNNClassifier(trainingSet_data, trainingSet_target, testSet_data, testSet_target):
    classChoice = int(input("1. Knn Class\n 2. Decision Tree \n >"))

    if classChoice == 2:
        myClassifyer = TreeClassifier()
    else:
        myClassifyer = Classifier()


    myClassifyer.train(trainingSet_data, trainingSet_target)

    # predictions of classifier
    predictionSet = myClassifyer.predict(testSet_data) # check

    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(trainingSet_data, trainingSet_target)
    predictions = classifier.predict(testSet_data)

    # calculate percent correct
    numCorrect = 0
    numGivenCorrect = 0
    for x in range(0, len(testSet_target)):
        if predictionSet[x] == testSet_target[x]:
            numCorrect += 1
        if predictions[x] == testSet_target[x]:
            numGivenCorrect += 1

    percentCorrect = numCorrect / float(len(predictionSet))
    percentGivenCorrect = numGivenCorrect / float(len(predictions))

    print ("Using my kNN percent correct is :")
    print("%.2f " %percentCorrect)
    print ("or ", numCorrect, " /", len(predictionSet))

    # for given classifier
    print ("Given kNN percent correct is :")
    print ("%.2f " %percentGivenCorrect)
    print ("or ", numGivenCorrect, " /", len(predictions))

    return


def runTreeTest():
    data = []
    targets = []
    labels = []
    choice = int(input("Choose Set: \n 1. Votes \n 2. Lenses \n 3. Iris \n> "))
    if (choice == 3):
        iris = datasets.load_iris()
        data = iris.data
        targets = iris.target
        data = convertToNom(data)
        labels = ["SepalL", "SepalW", "PetalL", "PetalW"]
    elif (choice == 2):
        lensesData = readLensData("lenses.csv")
        np_data = np.array(lensesData)
        data = np_data[:, :(len(lensesData[0]) - 1)]
        targets = np_data[:, (len(lensesData[0]) - 1)] # last column
        labels = ["Age", "Precription", "Astigmatic", "TearProduction"]
    else:
        votesData = readVotesData("votes.csv")
        np_data = np.array(votesData)
        data = np_data[:, -(len(votesData[0]) - 1):] #5 digits from the left
        targets = np_data[:, 0] # first column republican1 or democrate 0
        labels = ["handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution",
                    "physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban",
                    "aid-to-nicaraguan-contras","mx-missile","immigration","synfuels-corporation-cutback","education-spending",
                    "superfund-right-to-sue","crime","duty-free-exports","export-administration-act-south-africa"]


     # randomize
    np.unique(targets)
    np.random.seed(0)

    # 3 randomize the index
    indices = np.random.permutation(len(data))

    testNum = int(len(data) * 0.3)  # 70% train, 30% test

    trainingSet_data = data[indices[:-int(testNum)]]
    trainingSet_target = targets[indices[:-int(testNum)]]

    testSet_data = data[indices[-int(testNum):]]
    testSet_target = targets[indices[-int(testNum):]]

    train_classes = []
    test_classes = []
    # convert to arrays
    for t in trainingSet_target:
        train_classes.append(t)
    for t in testSet_target:
        test_classes.append(t)

    #print("Train Data:\n", trainingSet_data, train_classes)
    #print("Test Data:\n", testSet_data, test_classes)

    classifier = TreeClassifier()
    train_labels = []
    train_labels = list(labels)
    classifier.train(trainingSet_data, train_classes, labels)

    myPredictions = classifier.predict(testSet_data, train_labels)


    # TODO compare exsiting implemetations

    # calculate percent correct
    numCorrect = 0
    numGivenCorrect = 0
    for x in range(0, len(testSet_target)):
        if myPredictions[x] == test_classes[x]:
            numCorrect += 1
        #if predictionSet[x] == testSet_target[x]:
         #   numGivenCorrect += 1

    percentCorrect = numCorrect / float(len(test_classes))
    #percentGivenCorrect = numGivenCorrect / float(len(predictions))

    print ("Using my decision tree :")
    print("%.2f " %percentCorrect)
    print ("or ", numCorrect, " /", len(test_classes))

    return


def convertToNom(data):
    wordMatrix = np.chararray((len(data), len(data[0])), itemsize = 4) # row, col
    for i in range(0, len(data[0])): # for col
        col = data[:,i]
        maxVal = max(col)
        minVal = min(col)
        middle = (maxVal + minVal) / 2
        mTop = (maxVal + middle) / 2
        mBottom = (middle + minVal) / 2
        for j in range(0, len(col)): # for each row
                 # convert the col
                 if(col[j] > mTop):
                     wordMatrix[j][i] = "high"
                 elif(col[j] > mBottom):
                     wordMatrix[j][i] = 'med'
                 else:
                     wordMatrix[j][i] = 'low'

    return wordMatrix

def main(argv):
    runTreeTest()


if __name__ == "__main__":
    main(sys.argv)
