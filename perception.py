#encoding=utf8

import numpy as np
import tensorflow as tf
import numpy as np

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100)) # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 构造一个线性模型
#
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in xrange(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(W), sess.run(b)

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]

def loadDataSet():

    X = np.random.randint(10,size=(5,4))
    # print X.shape
    Y = np.random.randint(2,size=5)
    X = np.array([[1,1],[-1,1],[1,-1],[-1,-1]])
    Y = np.array([1,1,1,0])
    return X,Y


def perception(X,Y):
    """

    :param X:
    :param Y:
    :return:
    """
    xNum = X.shape[0]
    dim = X.shape[1]
    alpha = 0.2
    w = np.full(dim,1)
    b = 0
    flag = True
    while flag:
        flag = False
        deltaW = np.zeros(dim)
        for i in range(xNum):
            oSum = np.sum(x[i]*w)
            oSum += b
            # print x[i]*w,oSum
            poutput = [1 if oSum>=0 else 0][0]
            for d in range(dim):
                deltaW[d] = deltaW[d] + (Y[i] - poutput)*alpha*x[i][d]
                b = b + poutput*alpha
        for d in range(dim):
            if deltaW[d]!=0:
                w[d] += deltaW[d]
                flag = True
    return w,b

def sigmoid(x):
    return 1.0 / (1 + np.exp(x))

def stocGradAscent(dataMatrix,classlabel):

    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(np.sum(dataMatrix[i]*weights))
        error = classlabel[i] - h
        weights = weights + alpha*error*dataMatrix[i]
    return weights

def gradAscent(dataMat,classLabel):
    """

    :param dataMat:
    :param classLabel:
    :return:
    """
    dataMatrix = np.mat(dataMat)
    labelMat = np.mat(classLabel).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat-h)
        weights = weights + alpha * dataMatrix.transpose()*error
    return weights

if __name__ == '__main__':
    x,y = loadDataSet()
    w,b = perception(x,y)
    print w,b
