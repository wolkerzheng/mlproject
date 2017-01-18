#encoding=utf8
__author__ = 'ZGD'

import numpy as np
import copy
from sklearn.model_selection import train_test_split
def loadData():
    """
    加载数据集
    :return:
    """
    dataSet = []
    labels = []
    with open('iris.data') as f:
        for line in f.readlines():
            tmp = line.strip().split(',')
            labels.append(tmp[-1])
            tmp = [float(k) for k in tmp[:-1]]
            dataSet.append(tmp)
    X = np.array(dataSet)
    Y = np.array(labels)
    # print X
    return X, Y
def KFoldCrossValidation(X,Y,kSize=10):
    """
    :param X:
    :param Y:
    :param kSize:
    :return:
    """
    print '---------------------------------------------------'
    dataNum = len(X)
    n1 = dataNum % 10
    n2 = dataNum //10
    current = 0
    stop =  0
    sum = 0.0
    for i in xrange(kSize):
        if i < n1:
            stop = current + n2 + 1
        else:
            stop = current + n2
        x_test,y_test = X[current:stop],Y[current:stop]
        x_train =  np.concatenate((X[0:current], X[stop:]))
        y_train = np.concatenate((Y[0:current], Y[stop:]))
        current = stop
        clf = DesicionTree()
        Dtree = clf.fit(x_train, y_train)
        # ShowTree.drawtree(Dtree, 'Tree10FCV.jpeg')
        predicted = clf.predict(Dtree, x_test)
        rightNum = 0
        for j in range(len(predicted)):
            if predicted[j] == y_test[j]:
                rightNum += 1
        # print '第%d轮预测正确数：%d,总预测数目：%d'%(i+1,rightNum, len(predicted))
        sum = sum + float(rightNum * 1.0 / len(predicted))
        # print '第%d轮准确率:%f' % (i+1,float(rightNum * 1.0 / len(predicted)))
    print "准确率为：%f"%(sum/kSize)
    print '10折交叉验证成功'
    print '---------------------------------------------------'
def mypostprune(X, Y):
    """

    :param X:
    :param Y:
    :return:
    """
    print '---------------------------------------------------'
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    # mmode = raw_input('mode:=')
    clf = DesicionTree(mode="GainInfo")
    print "before postprune:"
    # result = clf.predict(test_x)
    Dtree = clf.BuildTree(x_train, y_train)
    predicted = clf.predict(Dtree, x_test)
    rightNum = 0
    for i in xrange(len(predicted)):
        if predicted[i] == y_test[i]:
            rightNum += 1
    print rightNum, len(predicted)
    print '准确率:%f' % float(rightNum * 1.0 / len(predicted))
    print "after postprune:"
    clf.postprune(Dtree, x_test,  y_test)
    # ShowTree.drawtree(Dtree,'TreeCarID3.jpeg')
    predicted = clf.predict(Dtree, x_test)
    rightNum = 0
    for i in xrange(len(predicted)):
        if predicted[i] == y_test[i]:
            rightNum += 1
    print rightNum, len(predicted)

    print '准确率:%f' % float(rightNum * 1.0 / len(predicted))
    print "post success"
    print '---------------------------------------------------'
class   DesicionTreeNode:
    """
    定义树的节点
    """
    def __init__(self, col=-1, val=None,labelDict = None,results=None, tb=None, fb=None):
        self.col = col
        self.val = val
        self.results = results
        self.labelDict = labelDict
        self.tb = tb
        self.fb = fb

class DesicionTree:
    def __init__(self,mode='ID3'):
        self.mode = mode
        # self.FeatColList = []

    def calEntropy(self, y):
        '''
        功能：calEntropy用于计算香农熵 e=-sum(pi*log pi)
        参数：其中y为数组array
        输出：信息熵entropy
        '''
        n = len(y)
        labelCounts = {}
        for label in y:
            if label not in labelCounts.keys():
                labelCounts[label] = 1
            else:
                labelCounts[label] += 1
        entropy = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / n
            entropy -= prob * np.log2(prob)
        return entropy

    def majorityCnt(self, labellist):
        """
        参数:labellist是类标签，序列类型为list
        输出：返回labellist中出现次数最多的label
        """
        labelCount = {}
        for vote in labellist:
            if vote not in labelCount.keys():
                labelCount[vote] = 0
            labelCount[vote] += 1
        sortedClassCount = sorted(labelCount.iteritems(), key=lambda x: x[1], \
                                  reverse=True)
        return sortedClassCount[0][0]

    def testErro(self,tree,x_test,y_test):
        """

        """
        error = 0.0
        predicted = self.predict(tree,x_test)
        for i in range(len(x_test)):
            if predicted[i] != y_test[i]:
                error +=1
        return float(error)

    def testMajor(self,tree,x_test,y_test,labelList):
        """
        :param tree:
        :param x_test:
        :param y_test:
        :return:
        """
        ctree = copy.deepcopy(tree)
        ctree.tb ,ctree.fb = None,None
        ctree.results = self.majorityCnt(labelList)
        error = 0.0
        predicted = self.predict(ctree, x_test)
        for i in range(len(x_test)):
            if predicted[i] != y_test[i]:
                error += 1
        return float(error)

    def postprune(self,tree ,x_test,y_test):
        """
        :param tree:
        :param x_test:
        :param y_test:
        :return:
        """
        if tree.tb.results == None:
            self.postprune(tree.tb,x_test,y_test)
        if tree.fb.results == None:
            self.postprune(tree.fb,x_test,y_test)
        if tree.tb.results != None and tree.fb.results != None:
            # Build a combined dataset
            labelList = []
            for v, c in tree.tb.labelDict.items():
                for i in range(c):
                    labelList.append(v)
            for v, c in tree.fb.labelDict.items():
                for i in range(c):
                    labelList.append(v)
            if self.testErro(tree,x_test,y_test) >= self.testMajor(tree,x_test,y_test,labelList):
                tree.tb, tree.fb = None, None
                tree.results = self.majorityCnt(labelList)
                tree.labelDict = self.finalLabelDict(labelList)

    def finalLabelDict(self,labellist):
        """
        统计叶子中各个类别出现的次数
        :param labellist:
        :return: 字典，存储类别，以及他们的数量
        """
        labelCount = {}
        for vote in labellist:
            if vote not in labelCount.keys():
                labelCount[vote] = 0
            labelCount[vote] += 1

        return labelCount

    def BuildTree(self,X, Y,attribute):
        """
        递归生长决策树
        :param X:训练样例集
        :param Y:这棵树要预测的目标属性
        :param attribute:属性列表
        :return:
        """
        labelList = list(Y)
        if labelList.count(labelList[0]) == len(labelList):
            leaf = DesicionTreeNode()
            leaf.results = labelList[0]
            leaf.labelDict = {labelList[0]:len(labelList)}
            return leaf
        if len(attribute) == 0 or len(set(X[0])) == 1:
            leaf = DesicionTreeNode()
            leaf.results = self.majorityCnt(Y)
            leaf.labelDict = self.finalLabelDict(Y)
            return leaf

        root = DesicionTreeNode()
        bestFeat = self.chooseBestFeatureToSplit_GainInfo(X,Y)
        bestFeatIndex,bestFeatValue = bestFeat[0]
        attributeVal = attribute[bestFeatIndex]
        attribute = list(attribute)
        attribute.remove(attributeVal)
        attribute = tuple(attribute)
        TrueXSet,FalseXSet,TrueYSet,FalseYSet = bestFeat[1]
        root.col = int(attributeVal[-1])
        root.val = bestFeatValue
        root.tb = self.BuildTree(TrueXSet,TrueYSet,attribute)
        root.fb = self.BuildTree(FalseXSet,FalseYSet,attribute)
        return root

    def chooseBestFeatureToSplit_GainInfo(self, X, y):
        """
        ID3：根据信息增益来选择最佳分裂属性
        参数：X为特征，y为label
        功能：根据信息增益或者信息增益率来获取最好的划分特征
        输出：返回最好划分特征的下标，其值，和最佳划分的分割后数据集
        """
        if not isinstance(X,np.ndarray):
            X = np.array(X)
        numFeat = X.shape[1]   #获取列数目
        baseEntropy = self.calEntropy(y)
        bestSplit = 0.0
        best_idx = -1
        best_val = 0
        best_res = []
        best_set = None
        best_colval = None
        for i in range(numFeat):
            featlist = X[:, i]  # 得到第i个特征对应的特征列
            uniqueVals = set(featlist)
            curEntropy = 0.0
            splitInfo = 0.0
            tmp = 0
            for value in uniqueVals:
                curEntropy = 0.0
                sub_x1, sub_x2,sub_y1,sub_y2 = self.splitDataSet(X, y, i, value)
                prob = len(sub_y1) / float(len(y))  # 计算某个特征的某个值的概率
                curEntropy = prob * self.calEntropy(sub_y1)	+ (1-prob)*self.calEntropy(sub_y2)  # 迭代计算条件熵
                IG = baseEntropy - curEntropy
                if IG > bestSplit:
                    bestSplit = IG
                    best_idx = i
                    best_colval=[best_idx,value]
                    best_set = [sub_x1, sub_x2,sub_y1,sub_y2]
        best_res = [best_colval,best_set]

        return best_res

    def splitDataSet(self,  X, y, index,  value):
        """
        分割数据集
        :param X:
        :param y:
        :param index:
        :param value:
        :return:返回分割后的两个数据集
        """
        split_function = None
        if isinstance(value, int) or isinstance(value, float):
            split_function = lambda row: row[index] <= value
        else:
            split_function = lambda row: row[index] == value
        Xset1, Xset2,Yset1,Yset2 = [],[],[],[]
        # Divide the rows into two sets and return them
        for i in xrange(len(X)):
            if split_function(X[i]):
                xx = [w for w in range(len(X[i])) if w != index]
                Xset1.append(X[i,xx])
                Yset1.append(y[i])
            else:
                xx = [w for w in range(len(X[i])) if w != index]
                Xset2.append(X[i,xx])
                Yset2.append(y[i])
        return Xset1, Xset2,Yset1,Yset2

    def fit(self, X, y):
        """
        参数：X是特征，y是类标签

        输出：self本身
        """
        featureIndex  = tuple(['x'+str(i) for i in range(X.shape[1])])
        # print featureIndex
        return self.BuildTree(X,y,featureIndex)

    def classfy(self,tree,observation):
        if tree.results != None:
            return tree.results
        else:
            v = observation[int(tree.col)]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v <= tree.val:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.val:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return self.classfy( branch,observation)

    def predict(self,tree,observation):
        """
        :param tree:
        :param observation:
        :return:
        """
        if tree == None:
            raise Exception('未建立决策树')
            return None
        if len(observation) < 1:
            raise Exception('测试数据不对')
            return None
        if len(observation) == 1:
            return self.classfy(tree,observation[0])
        else:
            results = []

            for i in range(len(observation)):
                tmp = self.classfy(tree,observation[i])
                results.append(tmp)
            return results


if __name__ == '__main__':
    X,Y = loadData()

    # holdoutMethod(X, Y)
    KFoldCrossValidation(X, Y)