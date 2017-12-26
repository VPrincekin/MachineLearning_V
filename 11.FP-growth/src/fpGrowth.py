#coding = utf-8
from numpy import *
import treeNode
"""
基于数据构建FP树:
    1.遍历所有的数据集合，计算所有项的支持度。
    2.丢弃非频繁的项.
    3.基于支持度降序排序所有的项。
    4.所有数据集合按照得到的顺序重新整理。
    5.重新整理完成后，丢弃每个集合末尾非频繁的项。 
    6.读取每个集合插入FP树中，同时用一个头部链表数据结构维护不同集合的相同项。
    7.最终得到FP树
"""
def createTree(dataSet, minSup=1):
    """
    生成FP树
    :param dataSet: 数据集字典 {行，出现的次数}
    :param minSup:  最小支持度
    :return:    返回FP树
    """
    headerTable = {}
    for trans in dataSet:  #开始遍历整个数据集字典{行：出现的次数}
        for item in trans:  #开始对每行数据遍历，统计每一行中每个元素出现的总次数
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in list(headerTable.keys()):    #删除headerTable中元素不满足最小支持度的元素
        if headerTable[k] < minSup:
            del headerTable[k]
    freqItemSet = set(headerTable.keys())   #满足minSup的元素集合
    if len(freqItemSet) == 0:   #如果这个集合是空，就返回None
        return None, None
    for k in headerTable:   #格式化headerTable {元素：[元素次数，None]}
        headerTable[k] = [headerTable[k], None]
    #创建FP树，从空集合开始
    retTree = treeNode.treeNode('Null Set', 1, None)  # create tree
    for tranSet, count in dataSet.items():  #开始遍历数据集字典{行：出现的次数}
        localD = {}
        for item in tranSet: #开始遍历每一行中的元素，判断在不在freqItemSet中，如果在加入字典 localD{元素：元素次数}
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            #如果localD不为空，那么根据全局频率对每个事务中的元素进行排序。
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            #排序之后，开始对树进行填充
            updateTree(orderedItems, retTree, headerTable, count)

    return retTree, headerTable

def updateTree(items, inTree, headerTable, count):
    """
    该函数的目的是为了让FP树生长。
    首先测试事务中的第一个元素项是否作为子节点存在。如果存在的话，则更新该元素项的计数。如果不存在，则创建一个新的treeNode并将其作为一个子节点添加到树中。
    这时，头指针表也要更新以指向新的节点。更新头指针表需要调用函数updataHeader().
    :param items:   满足最小支持度的元素key的数组（从大到小的排序）
    :param inTree:  空的retTree对象
    :param headerTable: 头指针列表 {元素：[元素次数，treeNode]}
    :param count:   原数据集中每一行元素出现的次数
    :return:
    """
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode.treeNode(items[0], count, inTree)
        # 如果满足minSup的dist字典的value值第二位为null， 我们就设置该元素为 本节点对应的tree节点.
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        # 如果元素第二位不为null，我们就更新header节点
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        # 递归的调用，在items[0]的基础上，添加item0[1]做子节点， count只要循环的进行累计加和而已，统计出节点的最后的统计值。
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    """
    该函数用于更新头指针，确保节点连接指向树中该元素项的每一个实例。
    从头指针的nodeLink开始，一直沿着nodeLink直到到达链表末尾。这就是链表。
    :param nodeToTest:
    :param targetNode: Tree对象的子节点
    :return:
    """
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def loadSimpDat():
    # 自定义数据
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    # 简单的数据包装器
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

if __name__ == '__main__':
    #测试生成FP树
    myDat = loadSimpDat()
    print(myDat)
    myDic = createInitSet(myDat)
    print(myDic)
    myFPtree,myHeaderTab = createTree(myDic,3)
    myFPtree.disp()