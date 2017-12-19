#coding=utf-8
from numpy import *
"""
CART算法的实现代码
函数creteTree()为代码大致如下：

    找到最佳的待切分特征:
        如果该节点不能再分，将该节点存为叶节点
        执行二元切分
        在右子树调用 createTree() 方法
        在左子树调用 createTree() 方法
"""
def loadDataSet(fileName):
    """
    解析文件数据函数
    :param fileName: 文件名
    :return:
    """
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        #将每行映射成浮点数
        fltLine = [float(x) for x in curLine]
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    """
    通过数组过滤方式将数据集合切分得到两个子集并返回。
    :param dataSet: 数据集合
    :param feature: 待切分的特征
    :param value:   待比较的特征值
    :return:
    """
    # nonzero(dataSet[:, feature] > value)  返回结果为true行的index下标
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

def regLeaf(dataSet):
    """
    负责生产叶节点，当确定不再对数据进行切分时，将调用该函数来得到叶节点的模型。在回归树种，该模型其实就是目标变量的均值。
    :param dataSet: 数据集
    :return:    最后一列的平均值
    """
    return mean(dataSet[:,-1])

def regErr(dataSet):
    """
    误差估计函数，计算总方差=方差*样本数
    :param dataSet: 数据集
    :return: 总方差
    """
    return var(dataSet[:,-1]) * shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    """
    回归树构建的核心函数，该函数的目的是找到数据的最佳二元切分方式。
    如果找不到一个‘好’的二元切分，该函数返回None并同时调用createTree()方法来产生叶节点，叶节点的值也将返回None.
    :param dataSet:     数据集
    :param leafType:    建立叶节点的函数
    :param errType:     误差计算函数（求总方差）
    :param ops:         [容许误差下降值,切分的最少样本数]
                        这个参数非常重要，因为它决定了决策树划分停止的threshold值，被称为预剪枝（prepruning），其实也就是用于控制函数的停止时机。
                        之所以这样说，是因为它防止决策树的过拟合，所以当误差的下降值小于tolS，或划分后的集合size小于tolN时，选择停止继续划分。
                        最小误差下降值，划分后的误差减小小于这个差值，就不用继续划分
    :return:    bestIndex 切分最优feature的index坐标
                bestValue 切分的最优值
    """
    tolS = ops[0]; tolN = ops[1]
    #取数据集的最后一列，转置为行向量，然后转换为list,该list是一个二维数组,只包含一个元素，就是最后一列所有值组成的list.
    distinctSet = set(dataSet[:,-1].T.tolist()[0])
    #如果数据集的最后一列所有值相等就退出
    if len(distinctSet) == 1:
        print(dataSet)
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    #无分类误差的总方差和
    S = errType(dataSet)
    #初始化最初总方差，最优特征index，最优特征值
    bestS = inf
    bestIndex = 0
    bestValue = 0
    #循环处理每一列
    for featIndex in range(n-1):
        #循环每一列的特征值
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):
            #按照当前列的当前特征值进行二元切分，返回两个子集
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            #判断二元切分返回的两个集合元素数量是否符合预期
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            #计算新的误差值，并与bestS误差比较，如果小于bestS，就说明当前列，当前特征值是最优。
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #判断二元切分方式的元素误差是否符合预期
    #如果误差减小不大则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    #对整体的成员进行判断，是否符合预期
    #如果切分出的数据集很小则退出
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    #最后如果这些提前终止条件都不满足，那么就返回切分特征和特征值。
    return bestIndex,bestValue

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    """
    树构建函数，一个递归函数。
    该函数首先尝试将数据集分成两个部分，切分由函数chooseBestSplit()完成。
    如果满足停止条件，将返回None和某类模型的值。如果构建的是回归树，该模型就是一个常数。如果是模型树，其模型就是一个线性方程。
    如果不满足停止条件，chooseBestSplit()会创建一个新的Python字典并将数据集分成两份，在这两份数据集上将分别继续递归调用createTree()函数。
    :param dataSet:     数据集
    :param leafType:    建立叶节点函数
    :param errType:     计算总误差函数
    :param ops:         [容许误差下降值,切分的最少样本数]
    :return:
    """
    #调用chooseBestSplit()函数，返回feature索引值，最优的feature值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    #如果chosseBestSplit()函数达到了某个停止条件，那么返回val.
    if feat == None: return val
    #创建一个字典来存储树
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    #按照最好的feature索引，以及最优的feature值来把数据集划分成两个集合，大于在右边，小于在左边
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    #分别在划分好的两个集合上递归调用createTree()函数，在左右子树中继续生成树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

if __name__ == '__main__':
    """#################################################################################"""
    #测试binSplitDataSe函数
    testMat = mat(eye(4))
    mat0,mat1 = binSplitDataSet(testMat,1,0.5)
    # print(mat0)
    # print(mat1)
    """##################################################################################"""
    #测试完整的createTree()函数
    myData = loadDataSet(r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\RegTrees\data\ex00.txt')
    myMat = mat(myData)
    resTree = createTree(myMat)
    print(resTree)
    #下面是多次切分的例子
    myData = loadDataSet(r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\RegTrees\data\ex0.txt')
    myMat = mat(myData)
    resTree = createTree(myMat)
    print(resTree)
    """
    到目前为止，已经完成回归树的构建，但是需要某种措施来检查构建过程是否得当，下面将介绍树剪枝技术，它通过对决策树剪枝来达到更好的预测效果。
    """