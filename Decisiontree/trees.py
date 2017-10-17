#coding=utf-8
import operator
#代码3-1，计算给定数据集的熵。
#创建数据集
def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels  
#计算给定数据集的熵
from math import log
def calcShannonEnt(dataSet):
    #计算数据集总样本个数
    numEntries = len(dataSet)
    #创建一个空字典
    labelCounts = {}
    #为所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    #计算数据集的熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        #以2为底求对数
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

#代码3-2，按照给定特征划分数据集。
#Args:待划分的数据集，划分数据集的特征(这里用的是索引)，需要返回的特征的值。
#Return:按照给定特征划分后的数据集。
def splitDataSet(dataSet,axis,value):
    #创建一个列表
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            #extend添加的是列表里面的元素，append添加的是整个列表。
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


#代码3-3，选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    #计算有几个特征feature
    numFeatures = len(dataSet[0])-1
    #计算给定数据集的原始熵，用于与划分完之后的数据集的熵值进行比较。
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;bestFeature = -1
    #开始遍历数据集中的所有特征。
    for i in range(numFeatures):
        #对每一组数据进行遍历
        featList = [example[i] for example in dataSet]
        #对得到的特征去重
        uniqueVals = set(featList)
        newEntropy = 0.0
        #遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集。然后计算数据集的新熵值。
        #并对所有唯一特征值得到的熵求和。
        for value in uniqueVals:
            #按照给定特征划分数据集
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        #最后，比较所有特征值中的信息增益，返回最好特征划分的索引值。
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
        return bestFeature
         
        
        
#代码3-4 多数表决法
"""该函数使用分类名称的列表，然后创建键值为classList中唯一值的数据字典，字典对象存储了classList中每个
    标签出现的频率，最后利用operator操作键值排序字典，并返回出现次数最多的分类名称。
"""
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    #字典的iteritems方法，返回一个迭代器。
    #operator提供的itemgetter函数用于获取对象的哪些维的数据。
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]



#代码3-5 创建树的函数代码
"""
Args:数据集，标签列表
Return: 树结构
"""
def createTree(dataSet,labels):
    #先获取所有类别
    classList = [example[-1] for example in dataSet]
    #类别完全相同则停止划分.count(统计某个元素在列表中出现的次数)
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #遍历完所有特征时返回出现次数最多的类别。
    if len(dataSet) == 1:
        return majorityCnt(classList)
    
    #选择最好的数据集划分方式.
    bestFeat = chooseBestFeatureToSplit(dataSet)
    #得到最好的特征
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    #删除
    del(labels[bestFeat])
    #得到最好划分方式的所有实例
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    #开始递归构建树
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    #返回myTree
    return myTree

