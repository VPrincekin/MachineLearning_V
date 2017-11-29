#coding=utf-8
"""
project：支持向量机
author:VPrincekin
"""

from numpy import *
def loadDataSet(fileName):
    """
    :param fileName: 文件名
    :return dataMat 特征列表---[[1,2],[3,4],....]
            labelMat 类别标签列表---[1,-1,....]
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    """
    返回一个0~m之间不为i的随机数
    :param i:
    :param m:
    :return: j
    """
    j=i
    while (j==i):
        j=int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    """
    :param aj: 目标值
    :param H: 最大值
    :param L: 最小值
    :return: 调整aj的值，返回aj
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def loadDataSet(fileName):
    dataMat=[]
    labelMat=[]
    fr = open(fileName)





