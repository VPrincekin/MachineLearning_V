#conding=utf-8
from numpy import *
import regression
"""
    如果数据的特征比样本点还多应该怎么办？是否还可以使用线性回归和之前的方法来做预测？
    答案是否定的，即我们不能再使用前面介绍的方法。这是因为在计算 矩阵求逆 的时候会出错。(这个地方不懂？？？)
    
1.为了解决这个问题，我们引入了岭回归（ridge regression），
简单来说，岭回归就是在原有的矩阵上加一个 λI 从而使得矩阵非奇异，
进而能对该矩阵求逆。其中矩阵I是一个 m * m 的单位矩阵， 对角线上元素全为1，其他元素全为0。

2.岭回归最先用来处理特征数多于样本数的情况，现在也用于在估计中加入偏差，从而得到更好的估计。
这里通过引入 λ 来限制了所有 w 之和，通过引入该惩罚项，能够减少不重要的参数，这个技术在统计学中也叫作 缩减(shrinkage)。

3.缩减方法可以去掉不重要的参数，因此能更好地理解数据。此外，与简单的线性回归相比，缩减法能取得更好的预测效果。
这里通过预测误差最小化得到 λ: 数据获取之后，首先抽一部分数据用于测试，剩余的作为训练集用于训练参数 w。
训练完毕后在测试集上测试预测性能。通过选取不同的 λ 来重复上述测试过程，最终得到一个使预测误差最小的 λ 。
"""

def ridgeRegres(xMat,yMat,lam=0.2):
    """
    经过岭回归公式计算得到的回归系数
    :param xMat: 样本的特征数据
    :param yMat: 类别标签
    :param lam:  引入的一个λ值，使得矩阵非奇异
    :return: 回归系数
    """
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws


def ridgeTest(xArr,yArr):
    """
    用于在一组 λ 上测试结果
    :param xArr: 样本的特征数据
    :param yArr: 类别标签
    :return: 将所有的回归系数输出到一个矩阵并换回
    """
    xMat = mat(xArr)
    yMat=mat(yArr).T
    #数据特标准化：所有的特征都减去各自的均值并除以方差
    #计算Y的均值
    yMean = mean(yMat,0)
    #Y的所有特征减去均值
    yMat = yMat - yMean
    #计算X的均值
    xMeans = mean(xMat,0)
    #计算X的方差
    xVar = var(xMat,0)
    #所有特征都减去各自的均值并除以方差
    xMat = (xMat - xMeans)/xVar
    #在 30 个不同的 lambda 下调用 ridgeRegres() 函数。
    numTestPts = 30
    #创建30 * m 的全部数据为0 的矩阵
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat
if __name__ == '__main__':
    """##############################################################################################"""
    #下面看一下数据集上的运行结果
    xArr,yArr = regression.loadDataSet(r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\Regression\data\abalone.txt')
    #得到30个不同lambda所对应的回归系数
    ridgeWeights = ridgeTest(xArr,yArr)
    #为了看到减缩效果，我们通过绘图显示
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()
    """
    上图绘制除了回归系数与 log(λ) 的关系。
        在最左边，即 λ 最小时，可以得到所有系数的原始值（与线性回归一致）；
        而在右边，系数全部缩减为0；在中间部分的某值将可以取得最好的预测效果。
    为了定量地找到最佳参数值，还需要进行交叉验证。另外，要判断哪些变量对结果预测最具有影响力，在上图中观察它们对应的系数大小就可以了。
    """