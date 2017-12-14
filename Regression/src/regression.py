#coding=utf-8
from numpy import *

def loadDataSet(fileName):
    """
    加载文件数据
    :param fileName:
    :return:
    """
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    """
    标准回归函数（计算ws,ws存放的就是回归系数。）
    :param xArr: 样本的特征数据
    :param yArr: 类别标签
    :return:  ws存放的就是回归系数
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T
    #计算x.T*x，判断它的行列式是否为0，如果为0，计算逆矩阵将会出现错误。
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    #有两种计算方式
    ws = xTx.I * (xMat.T*yMat)
    #ws = linalog.solve(xTx,xMat.T*yMat)
    return ws

def lwlr(testPoint,xArr,yArr,k=1.0):
    """
    局部加权线性回归函数(在待测点附近的每个点赋予一定的权重，在子集上基于最小均方差来进行普通的回归)
    :param testPoint: 样本点
    :param xArr: 样本的特征数据
    :param yArr: 类别标签
    :param k: 关于赋予权重矩阵的核函数的一个参数，与权重的衰减速率有关，可以人为调控。
    :return: testPoint * ws 数据点与具有权重系数相乘得到的预测点
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    #遍历整个数据集
    for j in range(m):
        #计算每个样本点对应的权重值：随着样本点与待预测点距离的递增，权重将以指数级衰减。
        diffMat = testPoint - xMat[j,:]
        #计算权重的公式（？）
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    """
     测试局部加权线性回归，对数据集中每个点调用 lwlr() 函数
    :param testArr: 测试数据集
    :param xArr: 样本的特征数据
    :param yArr: 类别标签
    :param k: 关于赋予权重矩阵的核函数的一个参数，与权重的衰减速率有关，可以人为调控。
    :return: 预测点的估计值
    """
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat



if __name__ == '__main__':
    #加载数据，利用standRegres()计算w
    xArr,yArr = loadDataSet(r"C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\Regression\data\ex0.txt")
    ws = standRegres(xArr,yArr)
    print(ws)
    #使用ws值计算Y
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat*ws
    #现在可以绘出数据集散点图和最佳拟合直线
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # scatter 的x是xMat中的第二列，y是yMat的第一列
    print(type(xMat[:, 1].flatten()),type(yMat.T[:, 0].flatten().A[0]))
    print(type(xMat[:, 1].tolist()), type(yMat.T[:, 0].flatten().A[0].tolist()))
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    #上面的代码创建了图像并绘出了原始的数据，为了绘制计算出最佳拟合直线，需要绘出yHat的值。
    #如果直线上的数据点次序混乱，绘图时将会出现问题，所以要将点a按照升序排列。
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()

    #我们可以通关计算预测值yHat序列和真实值y序列的匹配程度，那就是计算这两个序列的相关系数。
    #在numpy库中提供了相关系数的计算方法：可以通过corrcoef(yEstimate,yActual)来计算。
    corrArr = corrcoef(yHat.T,yMat)
    print(corrArr)

    #利用局部加权回归对单点进行估计
    yHat0 = lwlr(xArr[0],xArr,yArr,1.0)
    print(yArr[0],yHat0)
    yHat0 = lwlr(xArr[0], xArr, yArr, 0.1)
    print(yArr[0], yHat0)

    #为了得到数据集里所有点的估计，可以调用lwlrTest()函数
    yHat = lwlrTest(xArr,xArr,yArr,0.003)
    #下面绘出这些估计值和原始值，看看yHat的拟合效果。
    #所有的绘图函数需要将数据点按序排列，首先对xArr排序
    xMat = mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    #用Matplotlib绘图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='red')
    plt.show()
