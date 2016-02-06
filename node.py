import numpy as np
import itertools


class Node:
    branches = {} # list of other nodes for each branch
    labelName = ""
    resultClass = ""
    indexInLabelList = 0

    def __init__(self, labelValue = ""):
        self.branches = {}
        self.labelName = labelValue
        self.resultClass = ""
        return

    def appendBranch(self, Node):
        self.branches.append(Node)

    def isLeaf(self):
        if len(self.branches) == 0:
            return True
        else:
            return False