import numpy as np
import operator
from node import Node

class TreeClassifier:
    treeRoot = Node()
    all_data = []
    all_classes = []
    def __init__(self, data, classes): # constructor
        self.treeRoot = Node()
        self.all_data = data
        self.all_classes = classes
        return

    def calcEntropy(self, set):
        """
        Entropy measures the amount of impurity in a set
        1 = high means mixed set 0 = low
        (XXX YYY) - High entropy
        (xxx xxy) - low entropy = high info grain
        split the data (lhs & rhs)
        set1 = (x x x x x x x y y y y)
        H(Y) = - ∑ (p(yj) * log2(p(yj)))
        """
        tot = len(set)
        unique = np.unique(set)
        e = 0
        for x in unique:
            e += (set.count(x) / tot * np.log2(set.count(x)/tot))
        return e

    # helps with dictionary appending
    def insertIntoDataStruct(self, label, single_class, aDict):
        if not label in aDict:
            aDict[label] = [single_class]
        else:
            #print("SCLass:", single_class)
            aDict[label].append(single_class)

    # calculates the feature with the best info gain and returns the col
    def calcInfoGain(self, data, classes, currentLabel, currentIndex):
        result = 0.0
        valueFeq = {}
        numTotalItems = len(data)
        #print("Index", currentIndex, data)
        labelCol = data[:, currentIndex] # get the col we are testing

        for iRow in range(0,len(labelCol)): # for each row
            self.insertIntoDataStruct(labelCol[iRow], classes[iRow], valueFeq)

        #print("Current D: ", valueFeq)

        for key in valueFeq.keys(): # for every unique key
            numInSplit = len(valueFeq[key])
            result += (numInSplit/numTotalItems) * self.calcEntropy(valueFeq[key])
        #print("Returning gain for:", currentLabel, " is ", result)
        return result

    def  makeTree(self, data, classes, labels): # ID3 used here
        nData = len(data)
        numLabelsLeft = 0
        if labels:
            numLabelsLeft = len(labels)
        currentNode = Node()

        # the most common target
        #default = max(set(classes), key=classes.count)
        if(len(labels) == 0):
            print("Returning 1st if")
            #currentNode.resultClass = default
            return currentNode  # return a leaf node
        elif (len(classes) == 0):
            return currentNode
        elif (len(np.unique(classes)) == 1):
            print("Returning 1st if")
            currentNode.resultClass = classes[0]
            return currentNode
        else:
            # choose the best feature
            eList = np.zeros(numLabelsLeft)
            print("Num Labels left", numLabelsLeft)
            for ifeature in range(0, numLabelsLeft - 1): # the number features
                g = self.calcInfoGain(data, classes, labels[ifeature], ifeature)
                eList[ifeature] = g

            bestFeature = 0
            if eList.any():
                bestFeature = np.argmin(eList)
            print("BEST FEATURE, eList", bestFeature, eList)
            currentNode.labelName = labels[bestFeature]

           # print("Best Split:", labels[bestFeature])

            labels.remove(labels[bestFeature])
            #print("New Labels", labels)

            newDataDict = {}
            newClassDict = {}


            labelCol = data[:, bestFeature] # get the col we are testing
            possibleValues = np.unique(labelCol)
            for value in range(0, len(labelCol)): # for each row
                self.insertIntoDataStruct(labelCol[value], classes[value], newClassDict)
                self.insertIntoDataStruct(labelCol[value], data[value], newDataDict)

            #print("Send Classes:", newClassDict)


            # get the col we are testing
            # loop for the number of possible branches (i.e. low, med, high)
            print("Labels, Bestfeature", labels, bestFeature)
            labels.remove(labels[bestFeature])
            for value in possibleValues:
                temp = np.array(newDataDict[value])
                print("Send Data:", temp)
                currentNode.branches.append(self.makeTree(temp, newClassDict[value], labels))

        return currentNode


    def train(self, data, classes, labels):
        # modfies the root which is the start of the tree
        self.treeRoot = self.makeTree(data, classes, labels)
        return   # do nothing for now

    def predict(self, array): # array of data without classes

        return


