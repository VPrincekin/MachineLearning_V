import adaboost
from numpy import *
#coding=utf-8
from numpy import *

def loadDataSet(fileName):
    fr = open(fileName)
    lines = list(fr.readlines())
    linesLen = len(lines)
    print(linesLen)
    numFeat = len(lines[0].strip().split('\t'))
    dataMat = []
    labelMat = []
    for i in range(linesLen):
        lineArr = []
        curLine = lines[i].strip().split('\t')
        for j in range(numFeat-1):
            lineArr.append(float(curLine[j]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def loadDataSet2(fileName):
    fr = open(fileName)
    numFeat = len(fr.readline().split('\t'))
    dataArr = []
    labelArr = []
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataArr.append(lineArr)
        labelArr.append(float(curLine[-1]))
    return dataArr, labelArr

if __name__ == '__main__':
    dataMat,lableMat = loadDataSet('C:/Users/v_wangdehong/PycharmProjects/MachineLearning_V/AdaBoost/input_data/horseColicTraining2.txt')
    print(shape(dataMat),shape(lableMat))
    dataArr,labelArr = loadDataSet2('C:/Users/v_wangdehong/PycharmProjects/MachineLearning_V/AdaBoost/input_data/horseColicTraining2.txt')
    print(shape(dataArr),shape(labelArr))