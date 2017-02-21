#encoding=utf8

from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#加载数据集
iris = load_iris()
x = iris.data
y = iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


model = tree.DecisionTreeClassifier(criterion='gini')
# model = tree.DecisionTreeRegressor()

model.fit(x_train,y_train)

model.score(x_train,y_train)

predicted = model.predict(x_test)

accu = 0

for i in range(len(predicted)):
    if predicted[i] == y_test[i]:
        accu +=1

print '准确率:',accu/float(len(predicted))

