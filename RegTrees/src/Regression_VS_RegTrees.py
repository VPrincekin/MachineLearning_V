#coding=utf-8
from numpy import *
import model_Trees
import regTrees
import pruning_Trees
"""
前面介绍了模型树、回归树和一般的回归方法，下面测试一下哪个模型最好。
这些模型将在某个数据上进行测试，该数据涉及人的智力水平和自行车的速度的关系。当然，数据是假的。
"""


def regTreeEval(model, inDat):
    """
    回归树测试案例
    对于输入的单个数据点，或者行向量。返回一个浮点值。
    :param model:   输入模型
    :param inDat:   输入的测试数据
    :return:
    """
    return float(model)


def modelTreeEval(model, inDat):
    """
    模型树预测案例
    对输入数据进行格式化处理，在原数据矩阵上增加第0列，元素的值都是1，
    :param model:   输入模型
    :param inDat:   输入的测试数据
    :return:
    """
    n = shape(inDat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    """
    计算预测的结果
    在给定树结构的情况下，对于单个数据点，该函数会给出一个预测值。
    该函数自顶向下遍历整棵树，直到命中叶节点为止，一旦到达叶节点，它就会在输入数据上调用modelEval()函数，该函数的默认值为regTreeEval()
    :param tree:        已经训练好的树的模型
    :param inData:      输入的测试数据
    :param modelEval:   modelEval是对叶节点进行预测的函数引用，指定树的类型，以便在叶节点上调用合适的模型。可选值为 regTreeEval（回归树） 或 modelTreeEval（模型树），默认为回归树。
    :return:            返回预测值
    """
    if not pruning_Trees.isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if pruning_Trees.isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if pruning_Trees.isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval=regTreeEval):
    """
    多次调用treeForeCast()函数，对特定模型的树进行预测，可以是 回归树 也可以是 模型树。
    :param tree:        已经训练好的树的模型
    :param testData:    输入的测试数据
    :param modelEval:   预测的树的模型类型，可选值为 regTreeEval（回归树） 或 modelTreeEval（模型树），默认为回归树
    :return:            预测值矩阵
    """
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat

if __name__ == '__main__':
    # 加载数据集
    trainMat = mat(regTrees.loadDataSet(r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\RegTrees\data\bikeSpeedVsIq_train.txt'))
    testMat = mat(regTrees.loadDataSet(r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\RegTrees\data\bikeSpeedVsIq_test.txt'))

    # 创建一棵回归树测试
    myTree = regTrees.createTree(trainMat,ops=(1,20))
    yHat = createForeCast(myTree,testMat[:,0])
    corr1 = corrcoef(yHat,testMat[:,1],rowvar=0)
    print(corr1) #0.964

    # 创建一棵模型树测试
    myTree = regTrees.createTree(trainMat,model_Trees.modelLeaf,model_Trees.modelErr,ops=(1,20))
    yHat = createForeCast(myTree,testMat[:,0],modelTreeEval)
    corr2 = corrcoef(yHat,testMat[:,1],rowvar=0)
    print(corr2) #0.976

    # 使用标准线性回归测试
    ws,X,Y = model_Trees.linearSolve(trainMat)
    print(ws)
    for i in range(shape(testMat)[0]):
        yHat[i] = testMat[i,0]*ws[1,0]+ws[0,0]
    corr3 = corrcoef(yHat,testMat[:,1],rowvar=0)
    print(corr3) #0.943

    """
    从上面可以看出，模型树的预测效果是最好的，其次是回归树，最后是标准线性回归。
    """