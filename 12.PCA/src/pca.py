#coding = utf-8
from numpy import *
"""
在Numpy中实现PCA。
将数据转换成前N个主成分的伪代码：
    去除平均值
    计算协方差矩阵
    计算协方差举证的特征值和特征向量
    将特征值从大到小排序
    保留最上面的N个特征向量
    将数据转换到上述N个特征向量构建的新空间中
"""
def loadDataSet(fileName,delim='\t'):
    """
    加载数据函数
    :param fileName:
    :param delim:
    :return:
    """
    fr = open(fileName)
    datArr = []
    for line in fr.readlines():
        currLine = line.strip().split(delim)
        fltLine = [float(i) for i in currLine]
        datArr.append(fltLine)
    print(shape(datArr))
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):
    """
    PCA核心代码
    :param dataMat:     数据集
    :param topNfeat:    要应用的特征个数
    :return:
            lowDDataMat: 将维后的数据集
            reconMat:    新的数据集空间
    """
    meanVals = mean(dataMat, axis=0)   #计算每一列的均值
    meanRemoved = dataMat - meanVals    #每个向量同时都减去均值
    """
    cov协方差=[(x1-x均值)*(y1-y均值)+(x2-x均值)*(y2-y均值)+...+(xn-x均值)*(yn-y均值)]/(n-1)
    协方差：衡量两个变量的总体的误差。
        如果两个变量的变化趋势一致，也就是说如果其中一个大于自身的期望值，另外一个也大于自身的期望值，那么两个变量之间的协方差就是正值。 
        如果两个变量的变化趋势相反，即其中一个大于自身的期望值，另外一个却小于自身的期望值，那么两个变量之间的协方差就是负值。
    协方差矩阵：协方差矩阵的每个元素是各个向量元素之间的协方差。
    """
    covMat = cov(meanRemoved, rowvar=0) #计算协方差矩阵
    eigVals,eigVects = linalg.eig(mat(covMat))  # eigVals为特征值，eigVects为特征向量
    eigValInd = argsort(eigVals)    #对特征值进行从小到大的排序，返回从小到大的index号。
    eigValInd = eigValInd[:-(topNfeat+1):-1]   #特征值的逆序就可以得到topNfeat个最大的特征向量
    redEigVects = eigVects[:,eigValInd] #重组eigVects 最大到最小
    lowDDataMat = meanRemoved * redEigVects #将数据转换到新空间
    reconMat = (lowDDataMat * redEigVects.T) + meanVals #重构原始数据
    return lowDDataMat, reconMat

if __name__ == '__main__':
    #测试PCA函数
    dataMat = loadDataSet(r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\12.PCA\data\testSet.txt')
    print(shape(dataMat))
    lowDMat,reconMat = pca(dataMat,1)

    print(shape(lowDMat)) #将维后的数据集
    print(lowDMat)
    print(shape(reconMat)) #新的数据集空间
    print(reconMat)
    """
    我们可以将将维后的数据和原始数据一起绘制出来
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().tolist(),dataMat[:,1].flatten().tolist(),marker='^',s=90)
    ax.scatter(reconMat[:,0].flatten().tolist(),reconMat[:,1].flatten().tolist(),marker='o',s=50,c='red')
    plt.show()
