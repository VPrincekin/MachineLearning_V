#conding=utf-8
from numpy import *
import regression
"""
1.在增加如下约束时，普通的最小二乘法回归会得到与岭回归一样的公式
    ***限定了所有回归系数的平方和不能大于 λ .***
使用普通的最小二乘法回归在当两个或更多的特征相关时，可能会得到一个很大的正系数和一个很大的负系数。正式因为上述限制条件的存在，使用岭回归可以避免这个问题。

2.与岭回归类似，另一个缩减方法lasso也对回归系数做了限定，对应的约束条件如下:
    ***限定了所有回归系数的绝对值和不能大于 λ .***
唯一的不同点在于，这个约束条件使用绝对值取代了平方和。虽然约束形式只是稍作变化，结果却大相径庭: 在λ足够小的时候，一些系数会因此被迫缩减到 0.这个特性可以帮助我们更好地理解数据。

3.前向逐步回归算法可以得到与 lasso 差不多的效果，但更加简单。
它属于一种贪心算法，即每一步都尽可能减少误差。一开始，所有权重都设置为 1，然后每一步所做的决策是对某个权重增加或减少一个很小的值。
伪代码如下：

    数据标准化，使其分布满足0均值和单位方差 
    在每轮迭代过程中: 
    设置当前最小误差 lowestError 为正无穷
    对每个特征:
        增大或缩小:
            改变一个系数得到一个新的 w
            计算新 w 下的误差
            如果误差 Error 小于当前最小误差 lowestError: 设置 Wbest 等于当前的 W
        将 W 设置为新的 Wbest
"""
def regularize(xMat):
    """
    数据特征标准化
    :param xMat: 样本的特征数据
    :return: 归一化之后的样本特征数据
    """
    inMat = xMat.copy()
    inMeans = mean(inMat,0)#平均值
    inVar = var(inMat,0)#方差
    inMat = (inMat - inMeans)/inVar#每个特征减去平均值，除以方差
    return inMat

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    """
    前向逐步回归算法
    :param xArr: 样本的数据特征
    :param yArr: 类别标签
    :param eps: 每次迭代需要调整的步长
    :param numIt: 迭代次数
    :return:
    """
    xMat = mat(xArr)
    yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m,n=shape(xMat)
    #创建一个创建numIt* n的全部数据为0的矩阵
    returnMat = zeros((numIt,n))
    #创建一个n*1的向量来保存w的值
    ws = zeros((n,1))
    wsMax = ws.copy()
    #开始迭代
    for i in range(numIt):
        print (ws.T)
        lowestError = inf;
        #对每个特征进行循环
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                #改变一个系数得到一个新的w
                wsTest[j] += eps*sign
                #计算新w下的误差
                yTest = xMat*wsTest
                rssE = regression.rssError(yMat.A,yTest.A)
                #如果误差Error小于当前最小误差lowesError，这是wsTest等于当前W，否则的话不改变。
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

if __name__ == '__main__':
    """##########################################################################################################"""
    xArr, yArr = regression.loadDataSet(r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\Regression\data\abalone.txt')
    # print(stageWise(xArr,yArr,0.01,200))
    """
    1.从上面的输出结果中可以看到，w1和w6都是0，这表明他们不对目标值造成任何影响，也就是说着这些特征很可能不需要。
    2.另外，在参数设置为0.01的情况下，一段时间后系数就已经饱和并在特定值之间来回震荡，这是因为步长太大的缘故。这里会看到，第一个权重在0.04和0.05之间来回震荡。
    3.下面试着用更小的步长和更多的步数。
    """
    sw = stageWise(xArr,yArr,0.001,5000)
    print(sw)
    #同样可以绘图看到效果
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sw)
    plt.show()

    """##################################################################################################################"""
    #接着把这些结果与最小二乘法进行比较。
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xMat = regularize(xMat)
    yM = mean(yMat,0)
    yMat = yMat - yM
    # print(shape(xMat),type(xMat.tolist()),shape(xArr),type(xArr))
    # print(shape(yMat.T),type(yMat.T.tolist()),shape(yArr),type(yArr))
    ws = regression.standRegres(xMat,yMat.T)
    print(ws.T)
    """
    逐步线性回归算法的主要优点在于它可以帮助人们理解现有的模型并作出改进。
    当构建了一个模型后，可以运行该算法找出重要的特征，这样就有可能及时停止对那些不重要特征的收集。
    最后，如果用于测试，该算法每100次迭代后就可以构建出一个模型，可以使用类似于10折交叉验证的方法比较这些模型，最终选择使误差最小的模型。
    当应用缩减方法（如逐步线性回归或岭回归）时，模型也就增加了偏差，与此同时却减小了模型的方差。
    下面将通过LEGO的案例来看看偏差和方差间的折中效果。
    """
