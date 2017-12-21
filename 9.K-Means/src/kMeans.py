#coding=utf-8
from numpy import *
"""
簇: 所有数据点点集合，簇中的对象是相似的。
质心: 簇中所有点的中心（计算所有点的均值而来）.
SSE: Sum of Sqared Error（平方误差和）, SSE 值越小，表示越接近它们的质心. 由于对误差取了平方，因此更加注重那么远离中心的点.

K-Means伪代码：
    创建 k 个点作为起始质心（通常是随机选择）
    当任意一个点的簇分配结果发生改变时：
        对数据集中的每个数据点：
            对每个质心：
                计算质心与数据点之间的距离
            将数据点分配到距其最近的簇
        对每一个簇, 计算簇中所有点的均值并将均值作为质心
"""

def loadDataSet(fileName):
    """
    加载数据函数
    :param fileName: 文件名
    :return:   数据集
    """
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = [float(i) for i in curLine]
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    """
    计算两个向量的欧式距离(可根据场景选择)---根号下(坐标差的平方相加)
    :param vecA:
    :param vecB:
    :return:
    """
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataSet, k):
    """
    该函数为给定数据集构建一个包含K个随机质心的集合。
    随机质心必须要在整个数据集的边界之内，这可以通过找到数据集每一维的最小和最大值来完成。然后生成 0~1.0 之间的随机数并通过取值范围和最小值，以便确保随机点在数据的边界之内。
    :param dataSet:
    :param k:
    :return:
    """
    #获得列数
    n = shape(dataSet)[1]
    #创建一个k行，n列值都为0的矩阵
    centroids = mat(zeros((k, n)))
    #开始对每一列进行循环
    for j in range(n):
        #找到当前列的最小值
        minJ = min(dataSet[:, j])
        #取值范围等于当前列的最大值减去最小值
        rangeJ = float(max(dataSet[:, j]) - minJ)
        #随机生成（random.rand(k,1)表示生成k个0到1之间的随机数）
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """
    完整的K-均值算法。该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。
    这个过程重复数次，直到数据点的簇分配结果不再改变为止。
    :param dataSet:     数据集
    :param k:           k个质心
    :param distMeas:    计算距离的函数，可以改变，默认为distEclud.
    :param createCent:  为给定数据集构建一个包含K个随机质心的集合
    :return:
            centroids   k个质心的集合
            clusterAssment 簇分配的结果[最小质心的minIndex,最小距离minDist]
    """
    m = shape(dataSet)[0]
    #创建一个与dataSet行数一样，但是有两列的矩阵，用来保存簇分配的结果。
    clusterAssment = mat(zeros((m,2)))
    #创建包含k个随机质心的集合
    centroids = createCent(dataSet, k)
    #初始设定分簇结果发生改变进入循环
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        #开始遍历数据集每一个数据点并分配到最近的质心中去
        for i in range(m):
            #设定开始最小距离为正无穷
            minDist = inf
            minIndex = -1
            #开始遍历每个随机质心的集合
            for j in range(k):
                #计算数据点到质心的距离
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                #如果比最小距离小，就更新最小距离和最小质心的index。
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            #簇分配结果改变，更新簇分配结果为最小质心的minIndex,以及最小距离minDist。
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
                clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        #更新质心
        for cent in range(k):
            #获取发生改变簇中的所有点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            #将质心修改为簇中所有点的平均值
            centroids[cent,:] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment


if __name__ == '__main__':
    #测试randCent()函数,看看是否能生成min到max之间的值
    datDat = loadDataSet(r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\9.K-Means\data\testSet.txt')
    datMat = mat(datDat)
    centroids = randCent(datMat,2)
    # print(centroids)
    #测试一下距离计算方法
    dist = distEclud(datMat[0],datMat[1])
    # print(dist)
    #测试完整的K-均值算法
    myCentroids,myClusterAssment = kMeans(datMat,4)
    print(myCentroids,myClusterAssment)
    """
    上面的结果给出了4个质心。可以看到，经过3次迭代之后K-均值算法已经收敛。
    """