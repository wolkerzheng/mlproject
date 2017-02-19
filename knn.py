#encoding=utf8

import operator
import  numpy as np

class knn():

    def __init__(self,k=2):
        self.k = k

    def calculateDist(self,sample,dataset):
        """


        :param sample:
        :param dataset:
        :return:
        """
        m = dataset.shape[0]
        diffmat = np.tile(sample,(m,1)) - dataset
        sqDiffMat = diffmat**2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances**0.5
        return sorted(distances)


    def classfy0(self,sample,x,y):
        """

        :param sample:
        :param x:
        :param y:
        :return:
        """
        if isinstance(sample, np.ndarray) and isinstance(x, np.ndarray) \
                and isinstance(y, np.ndarray):
            pass
        else:
            try:
                sample = np.array(sample)
                x = np.array(x)
                y = np.array(y)
            except:
                raise TypeError("numpy.ndarray required for train_X and ..")
        sortedDistance = self.calculateDist(sample,x)
        classCount = {}
        for i in range(self.k):

            oneVote = y[sortedDistance[i]]
            classCount[oneVote] = classCount.get(oneVote,0) + 1
        sortedClass = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
        return sortedClass[0][0]

    def classify(self,test_x,train_x,train_y):
        """

        :param test_x:
        :param train_x:
        :param train_y:
        :return:
        """
        results = []
        if isinstance(test_x, np.ndarray) and isinstance(train_x, np.ndarray) \
                and isinstance(train_y, np.ndarray):
            pass
        else:
            try:
                test_X = np.array(test_x)
                train_X = np.array(train_x)
                train_y = np.array(train_y)
            except:
                raise TypeError("numpy.ndarray required for train_X and ..")
        d = len(np.shape(test_x))
        if d == 1:
            result = self.classfy0(test_x,train_x,train_y)
            results.append(result)
        else:
            for sample in test_x:
                result = self.classfy0(sample,train_x,train_y)
                results.append(result)
        return results


def image2vector(filename):

    returnVect = np.zeros((1,1024))

    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def readDataset():

    with open("./dataset/train-images.idx3-ubyte") as f:
        content = f.readlines()
        print len(content)
        # for line in content:
        #     print len(line),type(line)


    with open("./dataset/train-labels.idx1-ubyte") as f:
        content = f.readlines()
        print (content[0].split('\t'))
        # for line in content:
        #     print len(line),type(line)



if __name__ == '__main__':
    train_X = [[1, 2, 0, 1, 0],
               [0, 1, 1, 0, 1],
               [1, 0, 0, 0, 1],
               [2, 1, 1, 0, 1],
               [1, 1, 0, 1, 1]]
    train_y = [1, 0, 0, 0, 0]
    clf = knn(k=1)
    sample = [[1, 2, 0, 1, 0], [1, 2, 0, 1, 1],[1, 1, 0, 1, 1]]
    result = clf.classify(sample, train_X, train_y)
    print result