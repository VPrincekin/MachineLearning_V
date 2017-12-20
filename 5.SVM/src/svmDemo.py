#coding=utf-8

from numpy import *
from svmMLiA import *

def img2vector(filename):
    """
    将图像转化为测试向量
    :param filename: 文件名称（路径）
    :return: 向量
    """
    returnVector = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0,32*i+j] = int(lineStr[j])
    return returnVector

def loadImages(dirName):
    """
    这里对图像作了一个简单的二分类，除了1和9之外的的数字都被去掉了。
    :param dirName: 训练数据集文件名
    :return:
    """
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        #生成特征和类别标签
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


def testDigits(kTup=('rbf', 10)):
    """
    测试算法
    :param kTup: 核函数
    :return:
    """
    # 导入训练数据
    dataArr, labelArr = loadImages('C:/Users/v_wangdehong/PycharmProjects/MachineLearning_V/5.SVM/digits/trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    #构建支持向量矩阵
    #得到alphas矩阵中大于0的元素的位置，即得到所有支持向量的位置
    svInd = nonzero(alphas.A > 0)[0]
    #从数据中选出支持向量的特征
    sVs = datMat[svInd]
    #print(sVs)
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    # 2. 导入测试数据
    dataArr, labelArr = loadImages('C:/Users/v_wangdehong/PycharmProjects/MachineLearning_V/5.SVM/digits/testDigits')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))