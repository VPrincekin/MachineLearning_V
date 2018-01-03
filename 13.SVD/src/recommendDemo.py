#coding=utf-8
from numpy import *
from numpy import linalg as la
import svd
"""
推荐未尝过的菜肴
    推荐系统的工作过程：给定一个用户，系统会为此用户返回N个最好的推荐菜。
    实现流程大致如下：
        1·寻找用户没有评级的菜肴，即在用户-物品矩阵中的0值。
        2·在用户没有评级的所有物品中，对每个物品预计一个可能的评级分数。这就是说：我们认为用户可能会对物品的打分（这就是相似度计算的初衷）。
        3·对这些物品的评分从高到低进行排序，返回前N个物品。
[[1, 1, 0, 2, 2],
[0, 0, 0, 3, 3],
[0, 0, 0, 1, 1],
[1, 1, 1, 0, 0],
[2, 2, 2, 0, 0],
[5, 5, 5, 0, 0],
[1, 1, 1, 0, 0]]
"""
def standEst(dataMat, user, simMeas, item):
    """
    该函数用来在给定计算相似度方法的条件下，计算指定用户对物品的评分。
    :param dataMat:     数据矩阵
    :param user:        用户编号
    :param simMeas:    相似度计算方法
    :param item:        物品编号
    :return:
            用户对物品的评分
    """
    n = shape(dataMat)[1]   #获得数据矩阵的列数，n代表物品的种类数
    simTotal = 0.0; ratSimTotal = 0.0 # similarity  用户相似度，   userRating 用户评分
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0: #如果用户没有对该物品评分，跳过这个物品。
            continue
        #寻找两个用户都评级的物品，logical_and(,)计算两个元素的真值。overLap给出的是两个物品当中已经被评分的那个元素的索引ID。
        #也就是找到和该用户评级过同一件物品的用户，同时这些用户对该用户没有评级的物品也评级过。然后计算相似性评级。
        overLap = nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]
        if len(overLap) == 0: #如果两者没有任何重合元素，则相似度为0且终止此次循环
            similarity = 0
        else:   #如果存在重合的物品，则基于这些重合物品计算相似度。
            similarity = simMeas(dataMat[overLap,item], dataMat[overLap,j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        # 相似度会不断累加，每次计算时还考虑相似度和当前用户评分的乘积
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

def recommend(dataMat, user, N=3, simMeas=svd.cosSim, estMethod=standEst):
    """
    该函数会调用standEst(),最后产生最高的N个推荐结果
    :param dataMat:     数据矩阵
    :param user:        用户编号
    :param N:           最高的N个推荐结果
    :param simMeas:     计算相似度的函数
    :param estMethod:   估计方法
    :return:
    """
    #寻找未评级的物品，对给定的用户建立一个未评级的物品列表
    unratedItems = nonzero(dataMat[user,:].A==0)[1]

    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    #在未评分物品上进行循环
    for item in unratedItems:
        #估计方法，计算评分
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
        #按照估计得分，对该列表进行排序并返回。列表逆排序，第一个值就是最大值
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

if __name__ == '__main__':

    #行对应用户，列对应物品
    myMat = mat([[4,4,0,2,2],
                 [4,0,0,3,3],
                 [4,0,0,1,1],
                 [1,1,1,2,0],
                 [2,2,2,0,0],
                 [1,1,1,0,0],
                 [5,5,5,0,0]])
    #测试recommend函数。也可以尝试不同相似度计算函数
    print(recommend(myMat,2))
