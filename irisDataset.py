from sklearn import datasets
import itertools, random
import numpy as np
iris = datasets.load_iris()

def display(iris):
    # Show the data (the attributes of each instance)
    print(iris.data)
    # Show the target values (in numeric format) of each instance
    print(iris.target)
      # Show the actual target names that correspond to each number
    print(iris.target_names)

#display(iris)

class Classifier:
    def __init__(self): # constructor
        return

    def train(self, trainingSet_data, trainingSet_target):
        return   # do nothing for now

    def predict(self, array): # array of data without targets
        predictionArray = []
        # hardcoded returns the same class everytime
        for x in range(0, len(array)):# predicts a class and returns an array of predictions
            predictionArray.append(0)
        return predictionArray


#make a list of class instances
iris_data = iris.data
iris_target = iris.target
np.unique(iris_target)
np.random.seed(0)

# 3 randomize the index
indices = np.random.permutation(len(iris_data))

# 4 split the dataset in two sets training 70% and test 30%
trainingSet_data = iris_data[indices[:-45]]  # everything but the last 45 items
trainingSet_target = iris_target[indices[:-45]]

testSet_data = iris_data[indices[-45:]] # the last 45 items
testSet_target = iris_target[indices[-45:]]

# 5 - 6 train the Classifier
myClassifyer = Classifier()
myClassifyer.train(trainingSet_data, trainingSet_target)

#7 Determine predictions of classifier
predictionSet = myClassifyer.predict(testSet_data) # check
numCorrect = 0
for x in range(0, len(testSet_target)):
    if predictionSet[x] == testSet_target[x]:
        numCorrect += 1

percentCorrect = numCorrect / float(45)

print("Percent correct is :", percentCorrect, " or ", numCorrect, " / 45")