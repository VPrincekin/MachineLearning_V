#coding=utf-8
from numpy import *

def loadDataSet(fileName):
    """
    自适应数据加载函数
    :param fileName: 文件名
    :return:
    """
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

if __name__ == '__main__':
    import adaboost
    import adaboostDemo
    datMat, labelMat = adaboostDemo.loadDataSet('C:/Users/v_wangdehong/PycharmProjects/MachineLearning_V/6.AdaBoost/input_data/horseColicTraining2.txt')
    classifierArray = adaboost.adaBoostTrainDS(datMat, labelMat, 10)
    testArr, testLabelArr = adaboostDemo.loadDataSet('C:/Users/v_wangdehong/PycharmProjects/MachineLearning_V/6.AdaBoost/input_data/horseColicTest2.txt')
    prediction10 = adaboost.adaClassify(testArr, classifierArray)
    errArr = mat(ones((67, 1)))
    print(shape(mat(testLabelArr)),shape(prediction10))
    print(errArr[prediction10 != mat(testLabelArr).T].sum())

