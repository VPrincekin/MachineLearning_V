#coding=utf-8
#导入两个模块，第一个是科学计算包NumPy,第二个是运算符模块。
from numpy import *
import operator
from os import listdir
#创建数据集和标签
def createDataSet():
    #group矩阵每行包含一个不同的数据。
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    #向量labels包含来每个数据点的标签信息，labels包含的元素个数等于group矩阵行数
    labels=['A','A','B','B']
    return group,labels

#k-近邻算法
#inX:用于分类的输入向量.dataSet：输入的训练样本集.labels:标签向量。k:表示用于选择最邻近的数目。
def classify0(inX,dataSet,labels,k):
    #获取dataSet的矩阵行数
    dataSetSize = dataSet.shape[0]
    #距离计算。tile指定行重复，列重复。axis将每一行向量相加。
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #argsort()函数是将向量中的袁术从小到大排列，提取相应的index
    sortedDistIndicies=distances.argsort()
    
    #选择距离最小的个点
    classCount={}
    for i in range(k):
        #找到该样本的类型
        voteIlabel = labels[sortedDistIndicies[i]]
        #在字典中将该类型加一
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #排序并返回出现最多的那个类型
    #第一个参数：字典的iteritimes()方法，遍历返回字典的迭代器
    #第二个参数：这个参数的意思是先比较第几个元素，
    #reverse：代表是否倒序。
    sotedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sotedClassCount[0][0]


#示例：使用kNN算法改进约会网站的配对效果

#将文本记录转换为NumPy的解析程序
def file2matrix(filename):
    #从文件中读取数据
    fr = open(filename)
    arrayOLines = fr.readlines()
    #计算文件一共有多少行
    numberOfLines = len(arrayOLines)
    #生成对应的空矩阵，zero(n,m)就是生成一个n*m的矩阵，各个位置上都是0
    returnMat = zeros((numberOfLines,3))
    classLabelVector = [] 
    index = 0 
    for line in arrayOLines:
        line = line.strip()
        listFormLine = line.split('\t')
        returnMat[index,:] = listFormLine[0:3]
        classLabelVector.append(int(listFormLine[-1]))
        index+=1
    #返回knn训练算法需要的group,labels
    return returnMat,classLabelVector

#归一化特征值
def autoNorm(dataSet):
    #dataSet.min(0)中的参数0使得函数可以从列中选取最小值，而不是行的最小值。
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    #创建一个新的矩阵
    normDataSet = zeros(shape(dataSet))
    #dataSet的行数
    m = dataSet.shape[0]
    #矩阵中所有的值减去最小值
    normDataSet = dataSet - tile(minVals,(m,1))
    #矩阵中所有的值除以最大取值范围进行归一化
    normDataSet = normDataSet/tile(ranges,(m,1))
    #返回归一化矩阵，取值范围，每列最小值
    return normDataSet,ranges,minVals

#分类器针对约会网站的测试
def datingClassTest():
    #这个常亮的定义式为了截取10%的数据
    hoRatio = 0.10
    #调用处理数据的函数
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    #调用归一化特征值函数
    normMat,ranges,minVals = autoNorm(datingDataMat)
    #获取数据的行数
    m = normMat.shape[0]
    #取数据的10%
    numTestVecs = int(m*hoRatio)
    #定义错误计算变量
    errorCount = 0.0
    for i in range(numTestVecs):
        #调用classify0(indX，gropu,labels,k)对算法进行训练
        #normMat[i,:]代表第i条测试数据
        #numTestVecs:m 表示跳过前10%。
        #由此可以看出k-邻近算法的局限性，它不能自动优化，唯一可以调整的数值k还需要人工调整。
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d,the real answer is : %d" %(classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount+=1.0
    print("the total error rate is : %f" %(errorCount/float(numTestVecs)))

#约会网站预测函数
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person:",resultList[classifierResult - 1])
   
 #示例：手写识别系统
    
#准备数据：将图像转换为测试向量
def img2vector(filename):
    returnVector = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0,32*i+j] = int(lineStr[j])
    return returnVector
#测试算法：使用k-近邻算法识别手写数字
def handwritingClassTest():
    #定义一个列表
    hwLabels = [] 
    #获取训练数据集的文件名
    trainingFileList = listdir('trainingDigits')
    #计算训练数据集的样本个数
    m = len(trainingFileList)
    #准备把每个训练样本转换为一个（1，1024）的向量
    trainingMat = zeros((m,1024))
    #开始遍历训练样本
    for i in range(m):
        #得到每个图像对应的数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        #依次添加到列表中
        hwLabels.append(classNumStr)
        #得到每个图像对应的向量
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)

    #准备遍历测试数据
    testFileList = listdir('testDigits')
    errorCount = 0.0
    #获得测试数据个数
    mTest = len(testFileList)
    #开始遍历测试数据
    for i in range(mTest):
        #得到每个图像对应的数字
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        #得到每个图像对应的向量
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        #开始测试算法
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with: %d,the real answer is : %d" % (classifierResult,classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
        print("\nthe total number of errors is : %d" % errorCount)
        print("\nthe total error rate is : %f" % (errorCount/float(mTest)))
