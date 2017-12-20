#coding=utf-8
from numpy import *
import regTrees
"""
一棵树如果节点过多，表明该模型可能对数据进行了 “过拟合”。通过降低决策树的复杂度来避免过拟合的过程称为 剪枝（pruning）。
    在函数chooseBestSplit()中提前终止条件，实际上是在进行一种所谓的预剪枝（prepruning）操作。
    另一个形式的剪枝需要使用测试集和训练集,称作 后剪枝（postpruning）。
预剪枝：
    为了避免过拟合，可以设定一个阈值，熵减小的数量小于这个阈值，即使还可以继续降低熵，也停止继续创建分支。但是这种方法实际中的效果并不好。
后剪枝：
    决策树构造完成后进行剪枝。剪枝的过程是对拥有同样父节点的一组节点进行检查，判断如果将其合并，熵的增加量是否小于某一阈值。
    如果确实小，则这一组节点可以合并一个节点，其中包含了所有可能的结果。合并也被称作“塌陷处理” ，在回归树中一般采用取需要合并的所有子树的平均值。
    下面将讨论后剪枝，即利用测试集来对树进行剪枝，由于不需要用户指定参数，后剪枝是一种更理想化的剪枝方法
后剪枝伪代码：

    基于已有的树切分测试数据:
        如果存在任一子集是一棵树，则在该子集递归剪枝过程
        计算将当前两个叶节点合并后的误差
        计算不合并的误差
        如果合并会降低误差的话，就将叶节点合并
"""
def isTree(obj):
    """
    判断输入变量是否是一棵树，返回布尔类型的结果
    :param obj:
    :return:
    """
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    """
    一个递归函数，它从上往下遍历树直到叶节点为止。如果找到两个叶节点则计算他们的平均值。
    该函数对树进行塌陷处理，即返回树的平均值。
    :param tree: 输入的树
    :return:    返回树节点的平均值
    """
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):
    """
    该函数首先需要确认测试集是否为空。一旦非空，则反复递归调用函数prune()对测试数据进行切分。
    接下来要检查某个分支到底是子树还是节点。如果是子树，就调用函数prune()来对该子树进行剪枝。
    在对左右两个分支完成剪枝之后，还需要检查他们是否仍然还是子树，如果两个分支已经不再是子树，那么就可以进行合并。
    具体的做法是对合并前后的误差进行比较。如果合并后的误差比不合并的误差小就进行合并，否则的话不合并直接返回。
    :param tree: 待剪枝的树
    :param testData: 剪枝所需的测试数据
    :return:
    """
    if shape(testData)[0] == 0:
        return getMean(tree)
    # 判断分枝是否是dict字典，如果是就将测试数据集进行切分
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = regTrees.binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    # 如果是左边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    # 如果是右边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # 上面的一系列操作本质上就是将测试数据集按照训练完成的树拆分好，对应的值放到对应的节点

    # 如果左右两边同时都不是dict字典，也就是左右两边都是叶节点，而不是子树了，那么分割测试数据集。
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet =regTrees.binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        #计算总的误差
        #power(x,y)表示x的y次方
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
        #将两个分支合并并计算误差
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        #判断是否合并
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree

if __name__ == '__main__':
    #开始测试剪枝效果
    myDat = regTrees.loadDataSet(r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\RegTrees\data\ex2.txt')
    myMat = mat(myDat)
    myTree = regTrees.createTree(myMat,ops=(0,1))
    print(myTree)
    myDatTest = regTrees.loadDataSet(r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\RegTrees\data\ex2test.txt')
    myMatTest = mat(myDatTest)
    """
    可以看到，大量的节点已经被剪掉了，但没有预期的那样剪枝成两部分，这说明后剪枝可能不如预剪枝有效。
    一般地，为了寻求最佳模型可以同时使用两种剪枝技术。
    下面将重用部分已有的树构建代码来构建一种新的树。该树采用二元切分，但叶节点不再是简单的数值，取而代之的是一些线性模型。
    """
