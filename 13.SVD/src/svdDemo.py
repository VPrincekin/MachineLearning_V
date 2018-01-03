#coding=utf-8
from numpy import *
from numpy import linalg as la
import svd
import recommendDemo
"""
利用SVD提高推荐效果
"""
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def svdEst(dataMat, user, simMeas, item):
    """
    对数据集进行奇异值分解操作，利用U矩阵将数据转换到低维空间中。
    :param dataMat:
    :param user:
    :param simMeas:
    :param item:
    :return:
    """
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    #对数据集进行奇异值分解(SVD),在SVD分解之后，我们只利用包含了90%能量值的奇异值，这些奇异值会以NumPy数组的形式得以保存
    U,Sigma,VT = la.svd(dataMat)

    Sig4 = mat(eye(4)*Sigma[:4])
    #利用U矩阵将物品转换到低维空间中，构建转换后的物品(物品+4个主要的特征)
    xformedItems = dataMat.T * U[:,:4] * Sig4.I
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item:
            continue
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal


def printMat(inMat, thresh=0.8):
    """
    打印矩阵函数，由于矩阵包含了浮点数，因此必须定义浅色和深色。这里通过一个阈值来界定。
    :param inMat:
    :param thresh:
    :return:
    """
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1,end='')
            else: print(0,end='')
        print ('')

def imgCompress(numSV=3, thresh=0.8):
    """
    图像的压缩函数
    :param numSV:   给定的奇异值数目
    :param thresh:
    :return:
    """
    myl = []
    for line in open(r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\13.SVD\data\0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)
    #奇异值分解
    U,Sigma,VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):
        SigRecon[k,k] = Sigma[k]
    #重构图像
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)



if __name__ == '__main__':
    """
    测试奇异值分解预处理数据效果
    """
    myMat = mat(loadExData2())
    U,sigma,VT = la.svd(myMat)
    #接下来我们看看多少个奇异值能达到总能量的90%
    #首先计算总能量的90%
    sig2 = sigma**2
    print(sum(sig2)*0.9) #487.8
    #然后计算前两个元素所包含的能量
    print(sum(sig2[:2])) #378.829559511
    #从上面可以看出两个元素所包含的能量低于90%，于是计算前3个元素(当然这里也可以写个循环判断)
    print(sum(sig2[:3])) #500.500289128 这就说明我们可以将一个11维的矩阵转换成一个3维的矩阵。

    """
    测试svdEst()函数执行效果
    """
    N3=recommendDemo.recommend(myMat,1,estMethod=svdEst)
    print(N3)

    """
    测试基于SVD的图像压缩
    """
    imgCompress(2)