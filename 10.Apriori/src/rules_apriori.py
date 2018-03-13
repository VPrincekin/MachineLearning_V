#coding=utf-8
from numpy import *
import apriori
"""
前面已经提到利用关联分析可以发现许多有价值的信息。人们最常寻找的两个目标就是频繁项集与关联规则。
前面已经介绍了如何使用Apriori算法来发现频繁项集，现在需要解决的问题就是如何找出关联规则。
我们给出了频繁项集的量化定义，即它满足最小支持度要求。
对于关联规则，我们也有类似的量化方法，这种量化指标称之为 可信度。
一条规则 A -> B 的可信度定义为 support(A | B) / support(A)。（注意: 在 python 中 | 表示集合的并操作，而数学书集合并的符号是 U）。
A | B 是指所有出现在集合 A 或者集合 B 中的元素。
由于我们先前已经计算出所有 频繁项集 的支持度了，现在我们要做的只不过是提取这些数据做一次除法运算即可。

与我们前面的频繁项集生成一样，我们可以为每个频繁项集产生许多关联规则。如果能减少规则的数目来确保问题的可解析，那么计算起来就会好很多。
通过观察，我们可以知道，如果某条规则并不满足 最小可信度 要求，那么该规则的所有子集也不会满足 最小可信度 的要求。
假设 012 -> 3 并不满足最小可信度要求，那么就知道任何左部为 {0,1,2} 子集的规则也不会满足最小可信度的要求。 即 12 -> 03 , 02 -> 13 , 01 -> 23 , 2 -> 013, 1 -> 023, 0 -> 123 都不满足最小可信度要求。
可以利用关联规则的上述性质属性来减少需要测试的规则数目，跟先前 Apriori 算法的套路一样。
"""

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    """
    对规则进行评估：计算规则的可信度以及找到满足最小可信度要求的规则。
    :param freqSet:     频繁项集中的某个元素
    :param H:           频繁项集中的元素的集合。
    :param supportData: 所有元素的支持度的字典
    :param brl:         关联规则列表的空数组
    :param minConf:     最小可信度
    :return:
    """
    prunedH = [] #创建一个空集合来存储规则
    for conseq in H:
        #支持度定义: a -> b = support(a | b) / support(a).
        # 假设  freqSet = frozenset([1, 3]), conseq = [frozenset([1])]，
        # 那么 frozenset([1]) 至 frozenset([3]) 的可信度为
        # support(a | b) / support(a) = supportData[freqSet]/supportData[freqSet-conseq] =
        # supportData[frozenset([1, 3])] / supportData[frozenset([1])]
        conf = supportData[freqSet]/supportData[freqSet-conseq] #计算可信度
        if conf >= minConf: #判断规则的可信度符不符合要求
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    """
    该函数是为了从最初的频繁项集中生成更多的规则。
    :param freqSet:     频繁项集中的元素
    :param H:           频繁项集中的元素的集合。
    :param supportData: 所有元素的支持度的字典
    :param brl:         关联规则列表的空数组
    :param minConf:     最小可信度
    :return:
    """
    m = len(H[0])
    #先计算H中的频繁集大小m, 接下来查看频繁项集是否大到可以移除大小为m的子集。如果可以的话，则将其移除。
    if (len(freqSet) > (m + 1)):
        # 生成H中的无重复组合，这也是下一次迭代的H列表，Hmp1包含所有可能的规则
        Hmp1 = apriori.aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1): #如果不止一条规则满足要求，那么就使用Hmp1迭代调用rulesFromConseq()函数来判断是否进一步组合这些规则。
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    """
    主函数：生成关联规则，调用上面的两个函数
    :param L:   频繁项集列表
    :param supportData:     包含那些频繁项集和支持度的字典
    :param minConf:     最小的可信度阈值
    :return: 最后生成一个包含可信度的规则列表。
    """
    bigRuleList = []
    for i in range(1, len(L)): #遍历L中的每一个频繁项集。
        for freqSet in L[i]:    #每个频繁项集创建只包含单个元素集合的列表H1。
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1): #因为无法从单元素项集中构建关联规则，所以要从包含两个或者多个元素的项集开始构建规则。
                #生成候选规则集合
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                #对规则进行评估
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

if __name__ == '__main__':
    #测试利用Apriori生成关联规则
    dataSet = apriori.loadDataSet()
    #生成一个最小支持度是0.5的频繁项集的集合
    L,suppData = apriori.apriori(dataSet,0.5)
    print(L);
    print(suppData)
    #生成关联规则
    rules = generateRules(L,suppData)
    print(rules)
    """
    我们可以看到结果给出了3条规则，一旦降低可信度阈值，就会生成更多的规则。
    """