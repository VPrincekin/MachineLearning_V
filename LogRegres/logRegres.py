#coding=utf-8
from numpy import *

#代码5-1，Logistic回归梯度上升优化算法。
def loadDataSet():
    """解析文件
    Return: dataMat  文档列表 [[1,x1,x2]...]； labelMat 类别标签列表[1,0,1...]
    @author:VPrincekin
    """
    dataMat = []; labelMat= []
    fr = open('testSet.txt')
    #每行前两个分别是X1和X2，第三个只是数据对应的类别
    for line in fr.readlines():
        #strip()去除空格
        lineArr = line.strip().split()
        #为了方便计算，把X0设置为1。
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    """sigmoid函数
    @author:VPrincekin
    """
    return 1/(1+exp(-inX))

def gradAscent(dataMatIn,classLabels):
    """梯度上升算法
    Args: dataMatIn 文档矩阵 100*3 的矩阵；classLabels 类别标签列表 1*100向量
    Return: weights 回归系数矩阵
    @author:VPrincekin
    """ 
    #mat()转换为NumPy矩阵数据类型
    dataMatrix = mat(dataMatIn)
    #transpose()转置矩阵
    labelMat = mat(classLabels).transpose()
    #shape()求出矩阵的维度（行，列）
    m,n = shape(dataMatrix)
    #alpha 向目标移动的步长
    alpha = 0.001
    #maxCyles 迭代次数
    maxCycles = 500
    #创建一个n*1的单位矩阵
    weights = ones((n,1))
    #开始迭代,梯度上升
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights
     
######################################################################################

#代码5-2，画出数据集和Logistic回归最佳拟合直线的函数。
def plotBestFit(weights):
    """
    Args:weights 回归系数
    @author:VPrincekin
    """
    import matplotlib.pyplot as plt
    #解析文件，生成文档矩阵和类别标签矩阵
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    #开始画图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0,3.0,0.1)
    #此处设置了sigmoid函数为0，0是两个分类的分界处。w0x0+w1x1+w2x2=0
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
    
##############################################################################################

#代码5-3，随即梯度上升算法
def stocGradAscent0(dataMatrix,classLabels):
    """
    Args: dataMatrix 文档列表; classLabels 类别标签列表
    Return: weights 回归系数矩阵
    @author:VPrincekin
    """
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        #计算每一个样本的函数值
        h = sigmoid(sum(dataMatrix[i]*weights))
        #计算误差
        error = classLabels[i]-h
        #向梯度方向更新迭代
        weights = weights + alpha*error*dataMatrix[i]
    return weights

##############################################################################################

#代码5-4，改进的随即梯度上升算法
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    """
    Args:dataMatrix 文档列表; classLabels 类别标签列表; numIter 迭代次数，如果没有给定，默认迭代150次。
    Return:weights 回归系数矩阵
    @author:VPrincekin
    """
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter): 
        dataIndex = range(m)
        for i in range(m):
            #第一处改进，alpha在每次迭代的时候都会调整，这会缓解数据波动或者高频波动。
            alpha = 4/(1.0+i+j)+0.01
            #第二处改进，通过随机选取样本来更新回归系数。
            #这种方法将减少周期性波动，每次随即从列表中选出一个值，然后从列表中删掉该值。
            randIndex=int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
    return weights
            
########################################################################################################
            
#代码5-5，Logistic回归分类函数
def classifyVector(inX,weights):
    """测试算法
    Args: inX 测试样本; weigths 训练算法得到的回归系数
    Return: 返回类别，0或1.
    @author:VPrincekin
    """
    prob = sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0
    
def colicTest():
    """测试Logistic回归算法
    Args: None
    Return: Logistic回归算法错误率
    
    """
    #每个样本有21个特征，一个类别。
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    #开始解析训练文本，通过stocGradAscent1()计算并返回，回归系数向量。
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet),trainingLabels,500)
    #开始解析测试文本，计算算法的错误率。
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print('the error rata of this test is : %f' % errorRate)
    return errorRate

def multiTest():
    """调用colicTest()多次并求结果的平均值。
    @author:VPrincekin
    """
    numTests = 10; errorSum = 0.0 
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is : %f " %(numTests,errorSum/float(numTests)))
            