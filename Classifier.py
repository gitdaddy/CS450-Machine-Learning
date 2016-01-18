import numpy as np

class Classifier:

    def __init__(self): # constructor
        z_scores_np = []
        targetSet = []
        return

    def myKNN(self, unknown, k):
        result = 0
        dist_sq_array = []
        #print "current uknown: ", unknown
        for i in range (0, len(self.z_scores_np)):
            dist = 0 # reset the distance
            for j in range (0, len(unknown)):
                # acumulate the distance with all atributes
                dist += ((unknown[j] - self.z_scores_np[i][j])**2) # Eucildean distance^2

            temp = [dist, self.targetSet[i]]
            dist_sq_array.append(temp) # [0] = dist, [1] = target

        # get the shortest distance
        shortPair = []
        dist_sq_array = sorted(dist_sq_array) # sorts first col preserving dist - target association
        for i in range(0, k): # get k nearest neighbors
            shortPair.append(dist_sq_array[i])

        #print "Shortest distance pair{dist, target}:", shortPair

        targetRow = [row[1] for row in shortPair]
        # get the most common
        result = max(set(targetRow), key=targetRow.count)
        return result # return nearest target


    def train(self, trainingSet_data, trainingSet_target): # these indecies align
        self.targetSet = trainingSet_target
        # use columns for z_scores
        numRow = len(trainingSet_data)
        numCol = len(trainingSet_data[0])
        temp_z = np.zeros((numRow, numCol))
        for col in range(0, numCol):
            x_np = np.asarray(trainingSet_data[:,col])
            #print "Before: ", x_np
            tempZCol = (x_np - x_np.mean()) / x_np.std()
            #print "After Z score(adding):", tempZCol
            temp_z[:,col] = tempZCol # assign one column at a time
        self.z_scores_np = temp_z # save the scores
        return   # do nothing for now

    def predict(self, array): # array of data without targets
        predictionArray = []
        #print "Predict z scores: ", self.z_scores_np
        #print "Test array: ", array

        #standardize the test array
        z_test = (array - array.mean()) / array.std()
        #print "Z test: ", z_test
        for x in range(0, len(z_test)):# predicts a class using KNN
            predictionArray.append(self.myKNN(z_test[x], 5))

        return predictionArray


