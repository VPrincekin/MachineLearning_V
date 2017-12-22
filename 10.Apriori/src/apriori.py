#coding=utf-8
from numpy import *
"""
前面提到，关联分析的目标包括两项: 发现 频繁项集 和发现 关联规则。 首先需要找到 频繁项集，然后才能发现 关联规则。
Apriori 算法是发现 频繁项集 的一种方法。 Apriori 算法的两个输入参数分别是最小支持度和数据集。 该算法首先会生成所有单个物品的项集列表。 
接着扫描交易记录来查看哪些项集满足最小支持度要求，那些不满足最小支持度要求的集合会被去掉。 然后对剩下来的集合进行组合以生成包含两个元素的项集。
接下来再重新扫描交易记录，去掉不满足最小支持度的项集。 该过程重复进行直到所有项集被去掉。
"""
def loadDataSet():
    #创建了一个用于测试的简单数据集。
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    """
    构建集合C1，C1是大小为1的所有候选项集的集合。
    :param dataSet:
    :return:
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    #由于算法一开始是从输入数据中提取候选集列表，所以这里需要一个特殊的函数处理，而后续的项集列表则是按一定的格式存放的。
    #这里使用的格式是python中的frozenset,frozenset是指被‘冰冻’的集合，就是说它们不可改变的。
    #这里必须要使用forzenset而不是set类型，因为之后必须要将这些集合作为字典键值使用，而set做不到这一点。
    # return map(frozenset, C1)
    return [frozenset(i) for i in C1]

def scanD(D, Ck, minSupport):
    """
    该函数用于从C1生成L1。另外，该函数会返回一个包含支持度值得字典以备后用。
    :param D:  数据集
    :param Ck: 候选项集
    :param minSupport:  最小支持度（即数据集中包含该项集的记录所占的比例）
    :return:
            retList:    频繁项集列表
            supportData:    频繁项集列表的支持度
    """
    ssCnt = {} #创建一个空字典
    for tid in D: #遍历数据集中的所有交易记录
        for can in Ck: #遍历C1中的所有候选集。
            if can.issubset(tid):   #检测是否can中的每一个元素都在tid中。如果是的话则增加字典中对应的值
                # if not ssCnt.has_key(can):
                if not can in ssCnt.keys():
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D)) #数据集D的数量
    retList = []
    supportData = {}
    for key in ssCnt:  #遍历字典中的每个元素并且计算支持度。
        support = ssCnt[key] / numItems
        if support >= minSupport:   #如果获得的支持度大于最小支持度。
            retList.insert(0, key)  #在relist的首部插入指定元素。
        supportData[key] = support  #更新字典[元素，支持度]
    return retList, supportData


def aprioriGen(Lk, k):
    """
    aprioriGen（输入频繁项集列表 Lk 与返回的元素个数 k，然后输出候选项集 Ck。
    例如: 以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1}, {0,2}, {1,2}. 以 {0,1},{0,2},{1,2} 为输入且 k = 3 则输出 {0,1,2}
       仅需要计算一次，不需要将所有的结果计算出来，然后进行去重操作，这是一个更高效的算法
    :param Lk:   频繁项集列表
    :param k:    项集元素个数
    :return:    合并之后的频繁项集
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1==L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport = 0.5):
    """
    该函数首先构建集合 C1，然后扫描数据集来判断这些只有一个元素的项集是否满足最小支持度的要求。
    那么满足最小支持度要求的项集构成集合 L1。然后 L1 中的元素相互组合成 C2，
    C2 再进一步过滤变成 L2，然后以此类推，直到 CN 的长度为 0 时结束，即可找出所有频繁项集的支持度。
    :param dataSet:     数据集
    :param minSupport:  最小支持度
    :return:
                L: 频繁项集列表
                supportData: 包含那些频繁项集和支持度的字典
    """
    # C1 即对 dataSet 进行去重，排序，放入 list 中，然后转换所有的元素为 frozenset
    C1 = createC1(dataSet)
    # 对每一行进行 set 转换，然后存放到集合中
    D = [set(i) for i in dataSet]
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    #判断 L 的第 k-2 项的数据长度是否 > 0。
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k) # # 例如: 以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1}, {0,2}, {1,2}. 以 {0,1},{0,2},{1,2} 为输入且 k = 3 则输出 {0,1,2}
        Lk, supK = scanD(D, Ck, minSupport) #计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
        #保存所有候选项集的支持度，如果字典没有，就追加元素，如果有，就更新元素.
        supportData.update(supK)
        # Lk 表示满足频繁子项的集合，L 元素在增加，例如:
        # l=[[set(1), set(2), set(3)]]
        # l=[[set(1), set(2), set(3)], [set(1, 2), set(2, 3)]]
        L.append(Lk)
        k += 1
    return L, supportData


if __name__ == '__main__':
    #测试Apriori算法的辅助函数
    dataSet = loadDataSet()
    C1 = createC1(dataSet)
    print(C1) # [frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})]
    L1,supportData = scanD(dataSet,C1,0.5)
    print(L1) # [frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})]
    """
    通过输出结果可以看出.该列表中的每个单物品至少出现50%以上的记录中，由于物品4没有达到最小支持度，所以没有包含在L1中。
    通过去掉这件物品，减少了查找两件物品项集的工作量
    """
    #测试完整的Apriori算法
    L,suppData = apriori(dataSet)
    #此时的L包含满足支持度为0.5的频繁项集列表
    print(L[0])
    print(L[1])
    print(L[2])
    print(L[3])