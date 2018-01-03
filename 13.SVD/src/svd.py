#coding=utf-8
from numpy import *
from numpy import linalg as la
"""
利用Python实现SVD
"""
def loadExData():
    #生成简单的数据集
    return[[1, 1, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]


"""
相似度计算
"""
def ecclidSim(inA,inB):
    #欧式距离
    return 1.0/(1.0 + la.norm(inA - inB))

def pearsSim(inA,inB):
    #皮尔逊相关系数
    if len(inA)<3:
        return 1.0
    return 0.5 + 0.5*corrcoef(inA,inB,rowvar=0)[0][1]

def cosSim(inA,inB):
    #余弦距离
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5 + 0.5*(num/denom)



if __name__ == '__main__':
    """
    测试SVD效果
    """
    Data = loadExData()
    U,sigma,VT = la.svd(Data)
    print(sigma)#[  9.72140007e+00   5.29397912e+00   6.84226362e-01   5.88045416e-16   1.67039250e-16]
    #通过输出结果可以看出，前3个数值比其他的值大了很多，于是，我们就可以将最后两个值去掉。
    sig3 = mat([[sigma[0],0,0],[0,sigma[1],0],[0,0,sigma[2]]])
    #接下来，我们通过修改过后的矩阵来重构原始矩阵的相似矩阵。我们只需是用矩阵U的前3列和矩阵VT的前3行
    myMat = U[:,:3]*sig3*VT[:3,:]
    # print(myMat)

    """
    测试相似度计算
    """
    inA = mat(Data)[:,0]
    inB = mat(Data)[:,4]
    print(ecclidSim(inA,inA),ecclidSim(inA,inB))
    print(pearsSim(inA,inA),pearsSim(inA,inB))
    print(cosSim(inA,inA),cosSim(inA,inB))