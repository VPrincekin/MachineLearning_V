#coding = utf-8
from numpy import *
import treeNode
import fpGrowth
import excavate_fpGrowth
"""
有一个kosarak.dat文件，它包含将近100万条记录。该文件中的每一行包含某个用户浏览过的新闻报道。
一些用户只看过一篇报道，一些用户看过2498篇报道，用户和报道被编码成整数，所以查看频繁项集很难得到更多的东西，但是该数据对于展示FP-growth算法的速度十分有效。
"""
if __name__ == '__main__':
    #加载数据
    myDat = [line.split() for line in open(r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\11.FP-growth\data\kosarak.dat').readlines()]
    #初始化数据
    myDic = fpGrowth.createInitSet(myDat)
    #构建FP树,从中寻找那些至少被10万人浏览过的新闻报道
    myFPtree,myHeaderTab = fpGrowth.createTree(myDic,100000)
    #利用条件模式基递归查找频繁项集
    myFreqList = []
    excavate_fpGrowth.mineTree(myFPtree,myHeaderTab,100000,set([]),myFreqList)
    print(myFreqList)
    """
    总共有9个频繁项集
    """