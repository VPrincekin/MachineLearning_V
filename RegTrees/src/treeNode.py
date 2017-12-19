#conding=utf-8
from numpy import *
"""
CART 是十分著名且广泛记载的树构建算法，它使用二元切分来处理连续型变量。
即每次把数据集切分成两份。如果数据的某特征值等于切分所要求的值，那么这些数据就进入树的左子树，反之则进入树的右子树。
对 CART 稍作修改就可以处理回归问题，CART算法只做二元切分，所以这里可以固定树的数据结构:

    树包含左键和右键，可以存储另一棵子树或单个值。
    字典还包含特征和特征值这两个键，它们给出切分算法所有的特征和特征值。
"""
class treeNode():
    def __init__(self,feat,val,right,left):
        """
        树的结构
        :param feat:    特征
        :param val:     特征值
        :param right:   右子树
        :param left:    左子树
        """
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left

