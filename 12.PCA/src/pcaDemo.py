#coding = utf-8
from numpy import *
import pca
"""
半导体是在一些极为先进的工厂中制造出来的。设备的生命早期有限，并且花费极其巨大。
虽然通过早期测试和频繁测试来发现有瑕疵的产品，但仍有一些存在瑕疵的产品通过测试。
如果我们通过机器学习技术用于发现瑕疵产品，那么它就会为制造商节省大量的资金。

具体来讲，它拥有590个特征。我们看看能否对这些特征进行降维处理。

对于数据的缺失值的问题，我们有一些处理方法(参考第5章)
目前该章节处理的方案是：将缺失值NaN(Not a Number缩写)，全部用平均值来替代(如果用0来处理的策略就太差劲了)
"""
def replaceNanWithMean(datMat):
    """
    把数据集中所有的NAN替换为平均值
    :param datMat:  带有NaN的数据集
    :return:        替换后的数据集
    """
    numFeast = shape(datMat)[1]
    for i in range(numFeast):
        #对value不为NaN的求均值
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])
        #将value为NaN的值赋值为均值
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal
    return datMat

if __name__ == '__main__':
    #加载数据
    NonedatMat = pca.loadDataSet(r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\12.PCA\data\secom.data', ' ')
    #替换数据集中所有的NAN
    datMat = replaceNanWithMean(NonedatMat)
    #去除均值
    meanVals = mean(datMat,axis=0)
    meanRemoved = datMat - meanVals
    #计算协方差矩阵
    covMat = cov(meanRemoved,rowvar=0)
    #对该矩阵进行特征值分析
    eigVals,eigVects = linalg.eig(mat(covMat))
    print(eigVals)
    """
    我们会看到一大堆值，但是其中很多值都是0，这就意味着这些特征都是其他特征的副本，也就是说，它们可以通过其他特征表示，而本身没有提供额外信息。
    """
    lowDat,reconMat = pca.pca(datMat,40)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datMat[:, 0].flatten().tolist(), datMat[:, 1].flatten().tolist(), marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().tolist(), reconMat[:, 1].flatten().tolist(), marker='o', s=50, c='red')
    plt.show()
