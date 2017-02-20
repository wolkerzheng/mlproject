#encoding=utf8

import numpy as np

def loadDataSet():
    """
    构造加载数据集
    :return:
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive侮辱, 0 not
    return postingList, classVec

def createVocablist(dataset):
    """
    创建一个词库
    :param dataset:
    :return:
    """
    vocab = set([])
    for doc in dataset:
        vocab = vocab | set(doc)

    return list(vocab)

def setOfword2vec(vocab,inputset):
    """
    将单个样本映射到词库中,1表示出现该词
    :param vocab:
    :param inputset:
    :return:
    """
    returnVec = [0]*len(vocab)
    for w in inputset:
        if w in vocab:
            returnVec[vocab.index(w)] = 1
    return returnVec

def trainNB(trainMatrix,trainLabel):
    """
    输入：训练矩阵和类别标签,格式为numpy矩阵格式
        功能：计算条件概率和类标签概率
    :param trainMatrix:
    :param trainLabel:
    :return:
    """
    numTrainDocs = len(trainMatrix)     #统计样本个数
    numWords = len(trainMatrix[0])      #统计特征个数,词库的长度
    pNeg = sum(trainLabel)/float(numTrainDocs) #计算负样本出现的概率
    p0num = np.ones(numWords) #初始样本个数为1，防止条件概率为0，影响结果
    p1num = np.ones(numWords)

    p0inall = 2.0 #词库中只有两类，所以此处初始化为2(use laplace)
    p1inall = 2.0

    # 再单个文档和整个词库中更新正负样本数据
    for i in range(numTrainDocs):
        if trainLabel[i] ==1:
            p1num += trainMatrix[i]
            p1inall += sum(trainMatrix[i])

        else:
            p0num += trainMatrix[i]
            p0inall += sum(trainMatrix[i])

    # 计算给定类别的条件下，词汇表中单词出现的概率
    # 然后取log对数，解决条件概率乘积下溢
    p0Vect = np.log(p0num / p0inall)  # 计算类标签为0时的其它属性发生的条件概率
    p1Vect = np.log(p1num / p1inall)  # log函数默认以e为底  #p(ci|w=0)
    print p0Vect, p1Vect, pNeg
    return p0Vect, p1Vect, pNeg

def classfiNB(vecSample,p0vec,p1vec,pneg):
    """
     使用朴素贝叶斯进行分类,返回结果为0/1
    :param vecSample:
    :param p0vec:
    :param p1vec:
    :param pneg:
    :return:
    """
    prob_y0 = sum(vecSample * p0vec) + np.log(1-pneg)

    prob_y1 = sum(vecSample * p1vec) + np.log(pneg)  # log是以e为底
    if prob_y0 < prob_y1:
        return 1
    else:
        return 0

def testNb(sample):
    """
    测试
    :param sample:
    :return:
    """
    list0posts,listclassed = loadDataSet()
    myvocab = createVocablist(list0posts)
    trainMat = []

    for postinDoc in list0posts:
        trainMat.append(setOfword2vec(myvocab,postinDoc))
    p0v,p1v,pAb = trainNB(trainMat,listclassed)
    thisSample = np.array(setOfword2vec(myvocab,sample))
    result = classfiNB(thisSample,p0v,p1v,pAb)
    print sample,'classified as :',result
    return result

if __name__ == '__main__':

    testEntry = [['love', 'my', 'girl', 'friend'],
                 ['stupid', 'garbage'],
                 ['Haha', 'I', 'really', "Love", "You"],
                 ['This', 'is', "my", "dog"]]
    for item in testEntry:
        print testNb(item)