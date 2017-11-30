#coding=utf-8

from numpy import *

def img2vector(filename):
    """
    将图像转化为测试向量
    :param filename: 文件名称（路径）
    :return: 向量
    """
    returnVector = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0,32*i+j] = int(lineStr[j])
    return returnVector

