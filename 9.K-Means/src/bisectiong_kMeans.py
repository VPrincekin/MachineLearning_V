#coding=utf-8
from numpy import *
import kMeans
"""
聚类的目标是在保存簇数目不变的情况下提高簇的质量。一种度量聚类效果的指标是SSE(Sum ofSquared Error,误差平方和)
有两种可以量化的方法：合并最近的质心，或者合并使得SSE增幅最小的质心。
二分K-Means聚类算法：
    该算法首先将所有点作为一个簇，然后将该簇一分为二。
    之后选择其中一个簇继续进行划分，选择哪一个簇进行划分取决于对其划分时候可以最大程度降低 SSE（平方和误差）的值。
    上述基于 SSE 的划分过程不断重复，直到得到用户指定的簇数目为止。
另一种做法是选择 SSE 最大的簇进行划分，直到簇数目达到用户指定的数目位置。 接下来主要介绍该做法。
"""
def biKmeans(dataSet, k, distMeas=kMeans.distEclud):
    """
    给定数据集、期望的簇数目和距离计算方法的条件下，返回聚类结果。
    :param dataSet:     数据集
    :param k:           期望的簇数目
    :param distMeas:    距离计算方法
    :return:            聚类结果
    """
    m = shape(dataSet)[0]
    #创建一个矩阵，保存每个数据点的簇分配结果和平方误差。
    clusterAssment = mat(zeros((m,2)))
    #将质心初始化为所有数据点的均值。
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    #创建一个只有一个质心的list.
    centList =[centroid0]
    #计算每个数据点到质心的距离平方差
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    #判断当前簇数目是否达到预期
    while (len(centList) < k):
        #初始化最小SSE
        lowestSSE = inf
        #开始遍历每一个质心
        for i in range(len(centList)):
            #获取当前簇 i 下的所有数据点
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            #将当前簇 i 进行二分kMeans处理
            centroidMat, splitClustAss = kMeans.kMeans(ptsInCurrCluster, 2, distMeas)
            #将二分 kMeans 结果中的平方和的距离进行求和
            sseSplit = sum(splitClustAss[:,1])
            #将未参与二分 kMeans 分配结果中的平方和的距离进行求和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            #计算拆分后与未拆分时的误差和，误差和越小，划分的结果就越好。
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        #找出最好的簇分配结果？？？
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)   #当使用kMeans()函数并指定簇数为2时，会得到两个编号为0和1的结果簇。需要将这些簇编号修改为划分簇及新加簇的编号。
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit # 更新为最佳质心
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        #更新质心列表？？？
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]    #更新原质心 list 中的第 i 个质心为使用二分 kMeans 后 bestNewCents 的第一个质心
        centList.append(bestNewCents[1,:].tolist()[0])      # 添加 bestNewCents 的第二个质心
        # 重新分配最好簇下的数据（质心）以及SSE
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss
    return mat(centList), clusterAssment

if __name__ == '__main__':
    #测试二分K-Means聚类算法
    myDat = kMeans.loadDataSet(r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\9.K-Means\data\testSet2.txt')
    myMat = mat(myDat)
    centList,myNewAssments = biKmeans(myMat,3)
    print(centList)