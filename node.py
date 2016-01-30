import numpy as np

class Node:
    branches = [] # list of other nodes for each branch
    labelName = ""
    resultClass = ""
    def __init__(self, labelValue = ""):
        self.branches = []
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
