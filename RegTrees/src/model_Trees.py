#coding=utf-8
from numpy import *
import regTrees
"""
用树来对数据建模，除了把叶节点简单地设定为常数值之外，还有一种方法是把叶节点设定为分段线性函数，这里所谓的分段线性(piecewise linear)是指模型由多个线性片段组成。
如果使用两条直线拟合是否比使用一组常数来建模好呢？答案显而易见。决策树相比于其他机器学习算法的优势之一在于结果更易理解。很显然，两条直线比很多节点组成一棵大树更容易解释。
模型树的可解释性是它优于回归树的特点之一。另外，模型树也具有更高的预测准确度。将之前的回归树的代码稍作修改，就可以在叶节点生成线性模型而不是常数值。

下面将利用树生成算法对数据进行划分，且每份切分数据都能很容易被线性模型所表示。这个算法的关键在于误差的计算:
    那么为了找到最佳切分，应该怎样计算误差呢？前面用于回归树的误差计算方法这里不能再用。稍加变化，对于给定的数据集，应该先用模型来对它进行拟合.
    然后计算真实的目标值与模型预测值间的差值。最后将这些差值的平方求和就得到了所需的误差。
"""

def linearSolve(dataSet):
    """
    该函数主要是将数据集格式化成目标变量Y和自变量X。进行标准线性回归计算，可以得到回归系数ws.
    :param dataSet: 数据集
    :return:
            ws: 回归系数
            X ：格式化后的自变量X
            Y ：格式化后的目标变量Y
    """
    m,n = shape(dataSet)
    X = mat(ones((m,n)))
    Y = mat(ones((m,1)))
    #指定X的1到n列为数据集的0到n-1列，X的0列为1，代表常数项，用于计算平衡误差？
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]
    #转置矩阵*矩阵，和Regression中的回归系数计算公式一样
    xTx = X.T*X
    #如果矩阵的逆不存在，会造成程序异常
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):
    """
    该函数的功能与regTrees模块中的regLeaf()类似，负责生产叶节点，当确定不再对数据进行切分时，将调用该函数来得到叶节点的模型。
    不同的是，该函数在数据集上调用linearSolve()并返回回归系数ws
    :param dataSet: 数据集
    :return: ws: 回归系数
    """
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    """
    该函数可以在给定的数据集上计算误差。与regTrees模块中的regErr()类似，会被chooseBestSplit()调用来找到最佳切分。
    不同的是，该函数在数据集上调用linearSolve()并返回回归系数ws，X，Y。最后返回预测值和实际值之间的平方误差
    :param dataSet:
    :return:
    """
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))


if __name__ == '__main__':
    #测试模型树实际效果
    myDat = regTrees.loadDataSet(r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\RegTrees\data\exp2.txt')
    myMat = mat(myDat)
    myTree = regTrees.createTree(myMat,modelLeaf,modelErr,(1,10))
    print(myTree)
    """
    我们可以看到，该代码以0.285477为界创建了两个线性模型。
    模型树，回归树以及Regression里面的模型，哪一种模型更好呢？
    一个比较客观的方法是计算相关系数，也称为R^2值。该相关系数可以通过调用numpy库中的命令corrcoef(yHat,y,rowvar=0)来求解，其中yHat是预测值，y是真实值。
    """