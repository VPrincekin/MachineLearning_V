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
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree)  # if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):  # if the branches are not trees try to prune them
        lSet, rSet = regTrees.binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    # if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet =regTrees.binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree