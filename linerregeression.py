#encoding=utf8
"""
线性逻辑回归
"""
import math
import  numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.model_selection import train_test_split
#划分数据集
boston_sample,boston_target = load_boston(return_X_y=True)


x_train,x_test,y_train,y_test = train_test_split(boston_sample,boston_target,test_size=0.3)

#生成线性回归对象
liner = linear_model.LinearRegression()

#使用训练集生成模型


liner.fit(x_train,y_train)
liner.score(x_train,y_train)

print 'cofficient:',liner.coef_
print 'intercept:',liner.intercept_


predicted = liner.predict(x_test)


accu = 0
x = [i for i in range(1,len(predicted)+1)]
plt.plot(x,predicted,'r')
plt.plot(x,y_test,'b')

plt.show()

#假设预测时候差值为5算预测正确
# for i in range(len(predicted)):
#
#     if math.fabs(predicted[i]-y_test[i]) < 5:
#         accu +=1
#
# print '准确率:',accu/float(len(predicted))