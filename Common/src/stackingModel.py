#!/usr/bin/env python
#coding=utf-8
from importModule import *
from mlxtend.classifier import StackingClassifier
from sklearn import datasets


class StackModel():
    def __init__(self,clfArr,lr=linear_model.LogisticRegression()):
        """
        :param clfArr: 第一层想要融合的模型集合
        :param lr: 默认的第二层训练模型
        """
        self.clfArr = clfArr
        self.lr = lr

    def stack(self,X,y):
        """
        模型融合
        :param X: X是一个特征集合，array或者list
        :param y: Y是真实值集合，array或者list
        :return:
        """
        logging.info('------Stacking之后的模型效果')
        sclf = StackingClassifier(classifiers=self.clfArr,meta_classifier=self.lr)
        scores = model_selection.cross_val_score(sclf,X,y,cv=3,scoring='accuracy')
        print('The  Accuracy , mean: {:.2f} , std:+/- {:.2f}'.format(scores.mean(), scores.std()))

if __name__ == '__main__':
    iris = datasets.load_iris()
    x, y = iris.data[:, 1:3], iris.target
    clf1 = neighbors.KNeighborsClassifier(n_neighbors=1)
    clf2 = ensemble.RandomForestClassifier(random_state=1)
    clf3 = naive_bayes.GaussianNB()
    clfArr = list([clf1, clf2, clf3])
    sta = StackModel(clfArr)
    sta.stack(x, y)