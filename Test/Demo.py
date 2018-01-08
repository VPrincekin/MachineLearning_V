#coding=utf-8
from numpy import *
import json
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
import random as rnd

def loadData(fileName):
    fr = open(fileName)
    myData = []
    myLabel = []
    for line in fr.readlines():
        currLine = list(line.split('\t'))
        id = currLine[0]
        strVector = currLine[1].split(', ')
        one = (list(strVector[0])[1])
        mid = list(strVector[1:-1])
        end = list(strVector[-1])[0]
        mid.insert(0,one)
        mid.extend(end)
        vector = [float(i) for i in mid]
        if currLine[2].strip('\n')=='other':
            myLabel.append(float(1))
        else:
            myLabel.append(float(-1))
        myData.append(vector)
    return myData,myLabel

def demoPCA(dataMat, topNfeat=9999999):
    """
    PCA核心代码
    :param dataMat:     数据集
    :param topNfeat:    要应用的特征个数
    :return:
            lowDDataMat: 新的数据集空间
            reconMat:    降维后重构的数据集
    """
    meanVals = mean(dataMat, axis=0)   #计算每一列的均值
    meanRemoved = dataMat - meanVals    #每个向量同时都减去均值
    """
    cov协方差=[(x1-x均值)*(y1-y均值)+(x2-x均值)*(y2-y均值)+...+(xn-x均值)*(yn-y均值)]/(n-1)
    协方差：衡量两个变量的总体的误差。
        如果两个变量的变化趋势一致，也就是说如果其中一个大于自身的期望值，另外一个也大于自身的期望值，那么两个变量之间的协方差就是正值。 
        如果两个变量的变化趋势相反，即其中一个大于自身的期望值，另外一个却小于自身的期望值，那么两个变量之间的协方差就是负值。
    协方差矩阵：协方差矩阵的每个元素是各个向量元素之间的协方差。
    """
    covMat = cov(meanRemoved, rowvar=0) #计算协方差矩阵
    eigVals,eigVects = linalg.eig(mat(covMat))  # eigVals为特征值，eigVects为特征向量
    eigValInd = argsort(eigVals)    #对特征值进行从小到大的排序，返回从小到大的index号。
    eigValInd = eigValInd[:-(topNfeat+1):-1]   #特征值的逆序就可以得到topNfeat个最大的特征向量
    redEigVects = eigVects[:,eigValInd] #重组eigVects 最大到最小
    lowDDataMat = meanRemoved * redEigVects #将数据转换到新空间
    reconMat = (lowDDataMat * redEigVects.T) + meanVals #重构原始数据
    return lowDDataMat, reconMat

if __name__ == '__main__':
    myData,myLabel = loadData(r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\Test\traning.txt')
    myMat = mat(myData)
    myLabelMat = mat(myLabel).T
    # meanVals = mean(myMat,axis=0)
    # meanRemoved = myMat - meanVals
    # covMat = cov(meanRemoved,rowvar=0)
    # eigVals,eigVects = linalg.eig(mat(covMat))
    # lowDMat,reconMat = demoPCA(myMat,200)



    X_train = myMat[:650,:]
    X_test = myMat[650:,:]
    Y_train = myLabelMat[:650,:]
    Y_test = myLabelMat[650:,:]

    print(shape(X_train),shape(Y_train),shape(X_test),shape(Y_test)) #(685, 4798) (685, 1)

    # svc = SVC( C=1.0, kernel='sigmoid', degree=3, gamma='auto',
    #              coef0=0.0, shrinking=True, probability=False,
    #              tol=1e-3, cache_size=200, class_weight=None,
    #              verbose=False, max_iter=-1, decision_function_shape='ovr',
    #              random_state=None)
    # svc.fit(X_train,Y_train)
    # acc_svc = round(svc.score(X_train,Y_train)*100,2)
    # print(acc_svc)

    # knn = KNeighborsClassifier(n_neighbors=3)
    # knn.fit(X_train, Y_train)
    # acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
    # print(acc_knn)
    #

    ada = AdaBoostClassifier( n_estimators=100)
    ada.fit(X_train,Y_train)
    acc_ada = round(ada.score(X_test,Y_test)*100,2)
    print(acc_ada)




    #
    # ada = AdaBoostClassifier(n_estimators=10)
    # ada.fit(X_train, Y_train)
    # acc_ada = round(ada.score(X_train, Y_train) * 100, 2)
    # print(acc_ada)





    # print(shape(lowDMat),shape(reconMat))
    # for i in eigVals:
    #     print(i)