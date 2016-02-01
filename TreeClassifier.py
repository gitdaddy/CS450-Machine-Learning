import numpy as np
import operator
from node import Node

class TreeClassifier:
    treeRoot = Node()
    all_labels = []
    def __init__(self, labels): # constructor
        self.all_labels = labels
        self.treeRoot = Node()
        print("Class Labels:", self.all_labels)
        return

    def calcEntropy(self, set):
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
        #print("Entering with Labels:", labels)
        nData = len(data)
        numLabelsLeft = 0
        if labels:
            numLabelsLeft = len(labels)

        currentNode = Node()

        # the most common target
        default = max(set(classes), key=classes.count)
        if(numLabelsLeft == 0):
            #print("Returning no more labels")
            currentNode.resultClass = default
            currentNode.labelName = "End Node"
            return currentNode  # return a leaf node
        elif (len(classes) == 0):
            #print("Returning no more classes")
            return currentNode
        elif (len(np.unique(classes)) == 1):
            #print("Returning all data the same")
            currentNode.resultClass = classes[0]
            currentNode.labelName = labels[0]
            return currentNode
        else:
            # choose the best feature
            eList = np.zeros(numLabelsLeft)
            for ifeature in range(0, numLabelsLeft): # the number features
                g = self.calcInfoGain(data, classes, labels[ifeature], ifeature)
                eList[ifeature] = g

            #print("E List:", eList)
            bestFeature = 0
            if eList.any():
                bestFeature = np.argmax(eList)
            currentNode.labelName = labels[bestFeature]
            #currentNode.indexInLabelList = bestFeature # /~\

            labels.remove(labels[bestFeature])

            #print("Splitting on:", labels[bestFeature])

            newDataDict = {}
            newClassDict = {}

            labelCol = data[:, bestFeature] # get the col we are testing
            possibleValues = np.unique(labelCol)
            for value in range(0, len(labelCol)): # for each row
                self.insertIntoDataStruct(labelCol[value], classes[value], newClassDict)
                self.insertIntoDataStruct(labelCol[value], data[value], newDataDict)

            #print("Send Classes:", newClassDict)


            # get the col we are testing
            # loop for the number of possible branches (i.e. low, med, high) /~\
            #print("Labels, Bestfeature", labels, bestFeature)
            for value in possibleValues:
                temp = np.array(newDataDict[value])
                #print("Possible Value:", value)
                currentNode.branches[value] = self.makeTree(temp, newClassDict[value], labels)

        return currentNode

    def displayTree(self, root, newLevel, iteration):
        if root.isLeaf():
            print(" End target:", root.resultClass, end="")
        elif (root.branches) :
            print("  ", root.labelName, iteration, end="")
            if newLevel or iteration == 0:
                print("")
            for key in root.branches:
                iteration += 1
                if iteration == (len(root.branches) + 1):
                    self.displayTree(root.branches[key], False, iteration)
                else:
                    self.displayTree(root.branches[key], True, iteration)

        return

    def train(self, data, classes, labels):
        # modfies the root which is the start of the tree
        self.treeRoot = self.makeTree(data, classes, labels)

        #self.displayTree(self.treeRoot, False, 0)
        return   # do nothing for now

    def recurPredict(self, instance, root, labels):
        if root.isLeaf():
            return root.resultClass
        else:
            iCol = labels.index(root.labelName)
            for key in root.branches:
                if instance[iCol] == key:
                    return self.recurPredict(instance, root.branches[key], labels)


    def predict(self, array, labels): # array of data without classes
        result = []
        for instance in array:
            x = self.recurPredict(instance, self.treeRoot, labels)
            result.append(x)

        return result
