#coding=utf-8
from numpy import *
#Beautiful Soup是python的一个库，最主要的功能是从网页抓取数据.
from bs4 import BeautifulSoup
import regression
import ridge_regression
"""
用回归法预测乐高套装的价格
(1) 收集数据：用 Google Shopping 的API收集数据。（由于谷歌提供的api失效，数据已提前下载好，在data文件夹下的setHtml文件夹下。）
(2) 准备数据：从返回的JSON数据中抽取价格。
(3) 分析数据：可视化并观察数据。
(4) 训练算法：构建不同的模型，采用逐步线性回归和直接的线性回归模型。
(5) 测试算法：使用交叉验证来测试不同的模型，分析哪个效果最好。
(6) 使用算法：这次练习的目标就是生成数据模型。
"""

def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    """
    从页面读取数据，生成retX和retY列表
    :param retX:
    :param retY:
    :param inFile:
    :param yr:
    :param numPce:
    :param origPrc:
    :return:
    """''
    # 打开并读取HTML文件,指定解码格式，忽略非法字符
    fr = open(inFile,encoding='utf-8',errors='ignore')
    soup = BeautifulSoup(fr.read())
    i=1
    # 根据HTML页面结构进行解析
    currentRow = soup.findAll('table', r="%d" % i)
    while(len(currentRow)!=0):
        currentRow = soup.findAll('table', r="%d" % i)
        title = currentRow[0].findAll('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
        if len(soldUnicde)==0:
            print("item #%d did not sell" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].findAll('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','') #strips out $
            priceStr = priceStr.replace(',','') #strips out ,
            if len(soldPrice)>1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的套装价格
            if  sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        i += 1
        currentRow = soup.findAll('table', r="%d" % i)



def setDataCollect(retX, retY):
    """
    依次读取六种乐高套装的数据，并生成数据矩阵
    :param retX:
    :param retY: 
    :return: 
    """
    scrapePage(retX, retY, r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\Regression\data\setHtml\lego8288.html', 2006, 800, 49.99)
    scrapePage(retX, retY, r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\Regression\data\setHtml\lego10030.html', 2002, 3096, 269.99)
    scrapePage(retX, retY, r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\Regression\data\setHtml\lego10179.html', 2007, 5195, 499.99)
    scrapePage(retX, retY, r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\Regression\data\setHtml\lego10181.html', 2007, 3428, 199.99)
    scrapePage(retX, retY, r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\Regression\data\setHtml\lego10189.html', 2008, 5922, 299.99)
    scrapePage(retX, retY, r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\Regression\data\setHtml\lego10196.html', 2009, 3263, 249.99)


def crossValidation(xArr, yArr, numVal=10):
    """
    交叉验证测试岭回归
    :param xArr:    数据的特征集
    :param yArr:    类别标签
    :param numVal:  算计中交叉验证的次数。如果没有指定，默认是10.
    :return:
    """
    #获取数据点的个数
    m = len(yArr)
    indexList = arange(m)
    errorMat = zeros((numVal, 30))  # create error mat 30columns numVal rows
    #主循环，
    for i in range(numVal):
        #创建训练集和测试集的容器
        trainX = []
        trainY = []
        testX = []
        testY = []
        #使用numpy提供的shuffle函数对indexList中的元素进行混洗。
        #因此可以实现训练集或测试集数据点的随机选取。
        random.shuffle(indexList)
        #切分训练集和测试集
        for j in range(m):
            #创建一个基于数据集大小90%的训练集
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        #利用岭回归获得回归系数矩阵，得到30组回归系数组成的矩阵
        wMat = ridge_regression.ridgeTest(trainX, trainY)
        #循环遍历矩阵中的30组回归系数
        for k in range(30):
            #读取训练集和测试集
            matTestX = mat(testX)
            matTrainX = mat(trainX)
            #对数据进行标准化处理
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain
            #测试回归效果并存储
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)
            # yEst = matTestX * mat(wMat[k, :]).T
            errorMat[i, k] = regression.rssError(yEst.T.A, array(testY))
    #计算误差估计值得均值
    meanErrors = mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]
    #为了将得到的回归系数与standRegres()作对比，需要计算这些误差估计值的均值。
    #有一点值得注意，岭回归使用了数据标准化，而standRegres()没有，因此为了将上述比较可视化，还需将数据还原。
    xMat = mat(xArr)
    yMat = mat(yArr).T
    meanX = mean(xMat, 0)
    varX = var(xMat, 0)
    unReg = bestWeights / varX
    print("the best model from Ridge 7.Regression is:\n", unReg)
    print("with constant term: ", -1 * sum(multiply(meanX, unReg)) + mean(yMat))



if __name__ == '__main__':
    lgX = []
    lgY = []
    setDataCollect(lgX,lgY)
    crossValidation(lgX,lgY,10)
    #我们可以看一下在缩减过程中回归系数是如何变化的，我们可以通过不同的变化来分析哪些特征是关键的，哪些特征是不重要的。
    print(ridge_regression.ridgeTest(lgX,lgY))