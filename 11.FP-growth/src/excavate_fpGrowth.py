#coding=utf-8
from numpy import *
import treeNode
import fpGrowth
"""
从FP树种挖掘出频繁项集（三个基本步骤）
    1. 从FP树种获得条件模式基。（条件模式基是以所查找元素项为结尾的路径集合，每一条路径其实都是一条前缀路径。《左边路径，右边是值》）
    2. 利用条件模式基，构建一个条件FP树。
    3. 迭代重复步骤1，步骤2，直到树包含一个元素项为止。
"""
def ascendTree(leafNode, prefixPath):
    """
    迭代上溯整棵FP树，收集所有遇到的元素项的名称。
    :param leafNode:    要查询的节点对应的nodeTree
    :param prefixPath:  要查询的节点值
    :return:
    """
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):
    """
    为给定元素生成一个条件模式基。
    遍历链表直到结尾，每遇到一个元素项都会调用ascendTree来上溯FP树。
    :param basePat:     要查询的节点值
    :param treeNode:    查询的节点所在的当前nodeTree。
    :return:
    """
    condPats = {}
    #对treeNode的link进行循环
    while treeNode != None:
        prefixPath = []
        #寻找该节点的父节点，相当于找到了频繁项集。
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        # 递归，寻找改节点的下一个 相同值的链接节点
        treeNode = treeNode.nodeLink
    return condPats

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    """
    利用条件模式基递归查找频繁项集的minTree函数
    :param inTree:          事先创建好的FP树
    :param headerTable:     头指针列表
    :param minSup:          最小支持度
    :param preFix:          preFix为newFreqSet上一次的存储记录,初始为空。
    :param freqItemList:    用来存储频繁项集的列表
    :return:
    """
    #通过value进行从小到大的排序， 得到频繁项集的key组成的list。
    headerDic = {}
    for i in headerTable.keys():
        headerDic[i] = headerTable[i][0]
    # headerDic[i] = (headerTable[i][0] for i in headerTable.keys())
    bigL = [v[0] for v in sorted(headerDic.items(), key=lambda p: p[1])]
    for basePat in bigL: #循环遍历每个元素
        newFreqSet = preFix.copy()  #preFix为newFreqSet上一次的存储记录，一旦没有myHead，就不会更新
        newFreqSet.add(basePat)     #把每一个频繁项添加到频繁项集列表中。
        freqItemList.append(newFreqSet)
        #调用findPrefixPath()函数来创建条件模式基
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        #从条件模式基来构建FP树
        myCondTree, myHead = fpGrowth.createTree(condPattBases, minSup)
        #如果树种有元素项的话，递归调用mineTree()函数
        if myHead != None:
            print('conditional tree for:',newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

if __name__ == '__main__':
    # 测试给定元素项返回的条件模式基
    myDat = fpGrowth.loadSimpDat()
    myDic = fpGrowth.createInitSet(myDat)
    myFPtree, myHeaderTab = fpGrowth.createTree(myDic, 3)
    print(myHeaderTab)
    # condPatsX= findPrefixPath('x',myHeaderTab['x'][1])
    # print(condPatsX)
    # condPatsZ = findPrefixPath('z', myHeaderTab['z'][1])
    # print(condPatsZ)
    # condPatsR = findPrefixPath('r', myHeaderTab['r'][1])
    # print(condPatsR)

    # 测试利用条件模式基递归查找频繁项集
    freqItems = []
    myMinTree = mineTree(myFPtree,myHeaderTab,3,set([]),freqItems)
    print(freqItems)
    """
    正如我们所期望的那样，返回项集与条件FP树相匹配，到现在为止，完整的FP-growth算法以及可以运行。
    """