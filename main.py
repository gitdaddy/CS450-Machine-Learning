from sklearn import datasets
import itertools, random
import numpy as np
from io import StringIO
from Classifier import Classifier
from sklearn.neighbors import KNeighborsClassifier
from scipy import special, optimize
import pandas as pd
import csv


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
    with open('myCar.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for row in range(0, len(data)):
            spamwriter.writerow(data[row])

def testClassifier(trainingSet_data, trainingSet_target, testSet_data, testSet_target):

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

    print"Using my kNN percent correct is :"
    print("%.2f " %percentCorrect)
    print "or ", numCorrect, " /", len(predictionSet)

    # for given classifier
    print"Given kNN percent correct is :"
    print("%.2f " %percentGivenCorrect)
    print "or ", numGivenCorrect, " /", len(predictions)

    return

def runCar():

    carData = []
    carTargets = []
    with open('car.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in spamreader:
            carData.append(row)

    encodeCar(carData)
    np_data = np.array(carData)
    car_data = np_data[:, :6]
    car_targets = np_data[:, 6]

    # randomize
    np.unique(carTargets)
    np.random.seed(0)

    # 3 randomize the index
    indices = np.random.permutation(len(car_data))

    testNum = int(len(car_data) * 0.3)  # 70% train, 30% test
    print "Car Test Length is:",  testNum

    trainingSet_data = car_data[indices[:-int(testNum)]]
    trainingSet_target = car_targets[indices[:-int(testNum)]]

    testSet_data = car_data[indices[-int(testNum):]]
    testSet_target = car_targets[indices[-int(testNum):]]

    testClassifier(trainingSet_data, trainingSet_target, testSet_data, testSet_target)

    return

def runIris():
    iris = datasets.load_iris()

    #make a list of class instances
    iris_data = iris.data
    iris_target = iris.target
    np.unique(iris_target)
    np.random.seed(0)

    # 3 randomize the index
    indices = np.random.permutation(len(iris_data))

    print "Data length is:",  len(iris_data)
    testNum = (len(iris_data) * 0.3)  # 70% train, 30% test
    print "Test Length is:",  testNum

    trainingSet_data = iris_data[indices[:-int(testNum)]]  #
    trainingSet_target = iris_target[indices[:-int(testNum)]]

    testSet_data = iris_data[indices[-int(testNum):]]
    testSet_target = iris_target[indices[-int(testNum):]]

    testClassifier(trainingSet_data, trainingSet_target, testSet_data, testSet_target)

    return


def displaySet(set):
    # Show the data (the attributes of each instance)
    print(set.data)
    # Show the target values (in numeric format) of each instance
    print(set.target)
      # Show the actual target names that correspond to each number
    print(set.target_names)

""" from the book
def kNN(k, data, dataClass, inputs):
   nInputs = np.shape(inputs)[0]
   closest = np.zeros(nInputs)

   for n in range(nInputs):
       distances = np.sum((data-inputs[n,:])**2, axis=1)
       # identify the nearest neighbors
       indices = np.argsort(distances, axis=0)
       classes = np.unique(dataClass[indices[:k]])
       if(len(classes)== 1):
           closest[n] = np.unique(classes)
       else:
           counts = np.zeros(max(classes) + 1)
           for i in range(k):
               counts[dataClass[indices[i]]] += 1
           closest[n] = np.max(counts)"""

#START
choice = input("Choose Set: 1. Iris, 2. Car Set > ")
if (choice == 2):
    runCar()
else:
    runIris()

