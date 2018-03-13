#coding=utf-8
from numpy import *

def loadSimpData():
    """
    测试数据
    :return:
    """
    datMat = matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    """
    通过阈值比较对数据进行分类，所有在阈值一边的数据会分到类别1，而在另外一边的数据分到类别+1.
    :param dataMatrix: 数据集
    :param dimen: 特征列
    :param threshVal: 特征列要比较的值
    :param threshIneq: threshIneq == 'lt'表示修改左边的值，gt表示修改右边的值
    :return:
    """
    retArray = ones((shape(dataMatrix)[0],1))

    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    """
    遍历stumpClassify()函数所有的可能输入值，并找到数据集上最佳的单层决策树。
    :param dataArr: 数据集
    :param classLabels: 类别标签
    :param D:   最初的特征权重值
    :return:
            bestStump    最优的分类器模型
            minError     错误率
            bestClasEst  训练后的结果集
    """
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T

    m,n = shape(dataMatrix)
    #初始化数据
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m,1)))
    #初始化的最小误差为无穷大
    minError = inf
    #循环所有的feature列，将列切分成若干份，每一段以最左边的点作为分类点。
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        #根据最大值和最小值来计算步长
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):

            for inequal in ['lt', 'gt']:
                #如果是-1，那么得到rangeMin-stepSize; 如果是numSteps，那么得到rangeMax
                threshVal = (rangeMin + float(j) * stepSize)
                #对单层决策树进行简单分类，得到预测的分类值
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                #如果预测正确的话那么把对应的值改为0
                errArr[predictedVals==labelMat] = 0
                #将错误向量和权重向量的相应元素相乘并求和
                weightedError = D.T*errArr
                '''
                dim            表示 feature列
                threshVal      表示树的分界值
                inequal        表示计算树左右颠倒的错误率的情况
                weightedError  表示整体结果的错误率
                bestClasEst    预测的最优结果
                '''
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    # bestStump 表示分类器的结果，在第几个列上，用大于／小于比较，阈值是多少。
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    """
    完整的AdaBoost算法
    :param dataArr: 数据集
    :param classLabels: 类别标签
    :param numIt: 迭代次数
    :return:
    """
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    #建立一个列向量，记录每个数据点的类别估计累计值。
    aggClassEst = mat(zeros((m,1)))

    for i in range(numIt):
        #通过buidStump函数，返回的是具有最小错误率的单层决策树，同时返回的还有最小的错误率以及估计的类别向量。
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        classLabels = mat(classLabels)
        #计算每一个分类器实例的权重
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        #将alpha值加入到bestStump字典中
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print "classEst: ",classEst.T
        #计算下一次迭代的新权重向量D
        #分类正确：乘积为1，-1主要是下面求e的-alpha次方
        #分类错误：乘积为-1，结果会受影响，所以也乘以 -1
        expon = multiply(-1 * alpha *classLabels.T, classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        #预测的分类结果值，在上一轮结果的基础上进行加和操作
        aggClassEst += alpha*classEst
        #print(aggClassEst)
        #print "aggClassEst: ",aggClassEst.T
        #用sign判断正为1，负为-1，0为0。得到的是错误的样本标签集合
        aggErrors = multiply(sign(aggClassEst)!=classLabels.T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print("total error: ",errorRate)
        if errorRate == 0.0: break
    return weakClassArr

def adaClassify(datToClass,classifierArr):
    """
    利用训练出的多个弱分类器进行分类的函数
    :param datToClass: 一个或者多个到分类样例
    :param classifierArr: 多个弱分类器组成的数组
    :return:
    """
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    #建立一个列向量，记录每个数据点的类别估计累计值。
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        #基于stumpClassify函数对每个分类器的得到一个类别的估计值。
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        #输出估计值乘上该单层决策树的alpha权重然后累加到aggClassEst上。
        aggClassEst += classifierArr[i]['alpha']*classEst
        # print(aggClassEst)
    return sign(aggClassEst)