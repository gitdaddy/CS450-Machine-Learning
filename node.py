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

    def displayChildren(self):


        return

"""
    def block_width(self, block):
        try:
            return block.index('\n')
        except ValueError:
            return len(block)

    def stack_str_blocks(self,blocks):
        builder = []
        block_lens = [self.block_width(bl) for bl in blocks]
        split_blocks = [bl.split('\n') for bl in blocks]

        for line_list in itertools.zip_longest(*split_blocks, fillvalue=None):
            for i, line in enumerate(line_list):
                if line is None:
                    builder.append(' ' * block_lens[i])
                else:
                    builder.append(line)
                if i != len(line_list) - 1:
                    builder.append(' ')  # Padding
            builder.append('\n')

        return ''.join(builder[:-1])"""