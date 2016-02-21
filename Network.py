import numpy as np
from neuron import Neuron
import random
#import plotly.plotly as py
#import plotly.graph_objs as go
import matplotlib as plot
import bokeh as bokeh
from bokeh.plotting import figure, show, output_file, vplot


class Network_Layer:
    def __init__(self, layerInfo):
        # info: [4, 3, 3]
        self.layerInfo = layerInfo
        self.keySize = len(layerInfo) - 1
        isOutput = False
        self.layers = {} # dictionary containing the various nodes
        for i, item in enumerate(layerInfo):
            self.layers[i] = [] # create a spot for nodes where the key is just the indecies 0,1,2...n
            if (i + 1 == len(layerInfo)): # the output layer
                isOutput = True
            for j in range(0, item):
                # for layer[i] with the correct amount of wieghts + 1 = bias weight
                if(i == 0):
                    self.layers[i].append(Neuron( self.getRandList( layerInfo[i] + 1 ) ) )
                else:
                    self.layers[i].append(Neuron( self.getRandList( layerInfo[i-1] + 1 ), isOutput ) )

        return

    def getRandList(self, numItems):
        weightList = np.zeros(numItems) # one weight for every value plus bias list[0]
        # get a small random number for the weights
        for i in range(0, len(weightList)):
            SRN = random.uniform(-1, 1)
            weightList[i] = SRN
        return weightList

    def backPropagate(self, instance, target, isIris):
        """
        i j k
        output
        error(j) = (1 - a(j)) * (a(j) - t(j))
        hidden
        error(j) = (1 - a(j) * sum(j) * error(j)
        Update
        w(ij) = w(ij) - adda * error(j) * a(i)
        :param instance:
        :param target:
        :return:
        """
        learningRate = 0.1
        r = self.predict(instance)

        # using indecies as keys
        revKey = self.keySize
        tj = 0
        # gather all the error
        for i in range(0, self.keySize + 1): # for each layer using
            #print("in rev", "key:", revKey, self.layers[revKey])
            for j, node in  enumerate(self.layers[revKey]):
                if (node.outputLayer):
                    if (isIris):
                        if (target == j): # target as the correct ind
                            tj = 1
                        else:
                            tj = 0

                    node.error = (1 - node.lastOutput) * (node.lastOutput - tj)
                else:
                    sumError = self.sumError( self.layers[revKey + 1], j + 1)
                    node.lastOutput += 0.00001 # offset
                    node.error = (1.0 - node.lastOutput) * sumError # + 1 to skip the bias weight
                    #print("Error for layer:", revKey, " Node ", j, " last output: ", node.lastOutput, "Sum Error:", sumError, "Node Error:", node.error)

            revKey -= 1

        revKey = self.keySize
        # this may seem repetitive but there is a purpose
        for i in range(0, self.keySize + 1):
            for j, node in  enumerate(self.layers[revKey]):
                for x, weightij in enumerate(node.weights):

                    #print("Layer:", revKey, "Node:", j, " weight:", x)
                    if (x == 0): # bias wieght
                        node.weights[x] = self.getWeightUpdate(weightij, learningRate, node.error, -1)
                    elif (revKey == 0): # when i is the input layer
                        node.weights[x] = self.getWeightUpdate(weightij, learningRate, node.error, instance[x - 1])
                    else: # when exsits nodes behind
                        ai = self.layers[revKey - 1][x - 1].lastOutput
                        node.weights[x] = self.getWeightUpdate(weightij, learningRate, node.error, ai)


                #print("For layer:", revKey, " Node:", j, " New weights:", node.weights) #, " New weights:", node.newWeights)
            revKey -= 1

        return

    def getWeightUpdate(self, currentWieghtij, learningRate, dj, ai):
        #print("Current w(ij):", currentWieghtij, "Learning R:", learningRate, "error(j):", dj, "A(i):", ai)
        ret = currentWieghtij - (learningRate * dj * ai)
        #print("Result:", ret)
        return ret

    def sumError(self, layerk, nodeIndexj):
        sum = 0
        for nodek in layerk:
            sum += nodek.error * nodek.weights[nodeIndexj]
        return sum

    def train(self, train_data, train_classes, valid_data, valid_classes, isIris = False):
        """
        as it is training show a graph
        - acc on training = y
        - Epochs = x
        Max epochs var hardcoded
        above and beyound stop
        when validaiton set acc goes down
        :param train_data:
        :param train_classes:
        :return: a list of 1s and 0s
        """
        # set up graph
        output_file("line_graph.html")
        p = figure(plot_width=400, plot_height=400)

        # add a line renderer

        numEpochs = 15
        x_axis_points = []
        y_axis_points = []
        acc = 0
        stopLoop = False

        for i in range(0, numEpochs):
            if stopLoop:
                break
            x_axis_points.append(i)
            y_axis_points.append(acc * 100)
            print("ACC %.2f" %(acc), " Out of:", len(valid_classes))

            for i, instance in enumerate(train_data):
                if stopLoop:
                    break
                self.backPropagate(train_data[i], train_classes[i], isIris)
                # update graph using valid set
                acc = self.getAcc(valid_data, valid_classes, isIris)
                if (acc > .89): # break early
                    self.showGraph(p ,x_axis_points, y_axis_points, 2)
                    #print("trying to show graph")
                    stopLoop = True


        if(not stopLoop):
            self.showGraph(p ,x_axis_points, y_axis_points, 2)


        return

    def showGraph(self, p ,x_axis_points, y_axis_points, line_width):
        p.line(x_axis_points, y_axis_points, line_width=2)
        show(p)

    def getAcc(self, dataSet, targets, isIris = False):
        predictions =[]
        numCorrect = 0
        for i in range(0, len(dataSet)):
            if (isIris):
                predictions.append(np.argmax(self.predict(dataSet[i], True)))
            else:
                r = float(self.predict(dataSet[i]))
                predictions.append(int((r > 0.5))) # for pima float either close to 1 or 0

            if (predictions[i] == targets[i]):
                numCorrect += 1
        return float(numCorrect/len(targets))

    def predict(self, instance, isIris = False): # takes a single row and returns the results
     results = {}

     for key in self.layers:   # each layer
         results[key] = []

         for item in self.layers[key]: # for each node
            # take a single instance of the data for every node TEST
            # TEST
            if (key == 0):
                results[key].append(item.calcOutput(instance))
            else:
                results[key].append(item.calcOutput(results[(key - 1)])) # last results

     #print("Results :", results)
     if (isIris):
        return results[(len(self.layerInfo) - 1)] # return the outputs in a list
     else:
        return results[(len(self.layerInfo) - 1)][0] # return the outputs in a list

