#coding = utf-8
from numpy import *
"""
FP-growth:
    一种非常好的发现频繁项集算法。
    基于Apriori算法构建,但是数据结构不同，使用叫做 FP树 的数据结构结构来存储集合。
    基于数据构建FP树,从FP树种挖掘频繁项集
"""
class treeNode:
    """
    FP树结构
    """
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue   #节点名称
        self.count = numOccur   #节点出现次数
        self.nodeLink = None    #不同项集的相同项通过nodeLink连接在一起
        self.parent = parentNode    #指向父节点
        self.children = {}  #存储叶子节点

    def inc(self, numOccur):    #对count变量增加定值
        self.count += numOccur

    def disp(self, ind=1):  #将树以文本形式显示，对于树构建来说不是必要的，但是它对于调试非常有用。
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)

if __name__ == '__main__':
    #测试FP树所需数据结构
    rootNode = treeNode('pyramid',9,None)
    rootNode.children['eye'] = treeNode('eye',13,None)
    rootNode.children['phoenix'] = treeNode('phoenix',3,None)
    rootNode.disp()
