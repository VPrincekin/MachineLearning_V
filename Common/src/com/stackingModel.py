#!/usr/bin/env python
#coding=utf-8
from importModule import *
from mlxtend.classifier import StackingClassifier,StackingCVClassifier
from sklearn import datasets
"""
方法参数说明
    clfArr: type: List      第一层想要融合的模型组成的集合
    lr:     type: MLA       第二层的训练模型，默认为lr
    X:      type: List/array  训练数据集
    y:      type: List/array  类别标签
    text_X: type: List/array    测试数据集
"""

class StackModel():
    def __init__(self,clfArr,lr=linear_model.LogisticRegression()):
        """
        :param clfArr: 第一层想要融合的模型集合
        :param lr: 默认的第二层训练模型
        """
        self.clfArr = clfArr
        self.lr = lr

    def stack(self,X,y,test_X):
        """
        模型融合
        :param X: X是一个训练数据集合，array或者list
        :param y: Y是真实值集合，array或者list
        :param test_X: 测试数据集合，array或者list
        :return:
                result_Y：根据测试数据预测出来的结果
        """
        logging.info('------Stacking之后的模型效果')
        sclf = StackingCVClassifier(classifiers=self.clfArr,meta_classifier=self.lr,cv=4)
        # sclf = StackingClassifier(classifiers=self.clfArr,meta_classifier=self.lr,verbose=1)
        X=np.array(X)
        y=np.array(y).flatten()
        sclf.fit(X,y)
        result_Y = sclf.predict(test_X)
        scores = model_selection.cross_val_score(sclf,X,y,cv=5,scoring='accuracy')
        print('The  Accuracy , mean: {:.5f} , std:+/- {:.5f}'.format(scores.mean(), scores.std()))
        return result_Y

if __name__ == '__main__':
    """测试方法"""
    iris = datasets.load_iris()
    x, y = iris.data[:, 1:3], iris.target
    clf1 = neighbors.KNeighborsClassifier(n_neighbors=1)
    clf2 = ensemble.RandomForestClassifier(random_state=1)
    clf3 = naive_bayes.GaussianNB()
    clfArr = list([clf1, clf2, clf3])
    sta = StackModel(clfArr)
    sta.stack(x,y,x)