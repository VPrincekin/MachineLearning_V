#coding=utf-8
from numpy import *
import apriori
import rules_apriori
if __name__ == '__main__':
    myshDatSet =[line.split() for line in open(r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\10.Apriori\data\mushroom.dat').readlines()]
    L,suppData = apriori.apriori(myshDatSet,0.3)
    #可以在结果中搜索包含有毒特征值2的频繁项集
    for item in L[1]:
        if item.intersection('2'):
            print(item)
    # #当然也可以对更大的项集来重复上述过程
    for item in L[3]:
        if item.intersection('2'):
            print(item)