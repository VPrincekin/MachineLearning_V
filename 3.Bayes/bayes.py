#coding=utf-8

#代码4-1，词表到向量的转换函数
from numpy import *
def loadDataSet():
    """创建数据集
    Return:单词列表postingList,类别标签集合classVec
    """
    postingList=[['my','dog','has','flea','problems','help','please'],
             ['maybe','not','take','him','to','dot','park','stupid'],
             ['my','dalmation','is','so','cute','I','love','him'],
             ['sotp','posting','stupid','worthless','garbage'],
             ['mr','licks','ate','my','steak','how','to','stop','him'],
             ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    """获取所有单词的集合
    Args:dataSet 数据集
    Return:vocabSet所有单词的集合
    """
    #创建一个空的集合
    vocabSet = set([])
    for document in dataSet:
        #返回两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    """遍历文档中的所有单词，输出文档词向量，出现为1，未出现为0
    Args:vocabList 所有单词集合列表; inputSet 输入数据集
    Return:returnVec 文档匹配向量[0，0，1...]
    """
    #创建一个和词汇表长度一样的向量，全为0。
    returnVec = [0]*len(vocabList)
    #开始对输入的文本进行遍历。出现设置为1.
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else : print('the word: %s is not in my Vocablary' % word)
    return returnVec

###############################################################################

#代码4-2，朴素贝叶斯分类器训练函数
def _trainNB0(trainMatrix,trainCategory):
    """
    Args:trainMatrix 文档矩阵，由一组组文档向量构成[1,0,0,1...]； trainCategory 每篇文档类别标签所构成的向量 [1,0,1,0,1,0]
    Return:p0Vect 正常文档中单词占比列表；p1Vect 侮辱性文档中单词占比列表；pAubsive 任意一篇文档属于侮辱性文档的概率。
    """
    #得到文档一共有多少个样本文件
    numTrainDocs = len(trainMatrix)
    #得到每组词向量的长度
    numWords = len(trainMatrix[0])
    #得到任意文档属于侮辱性文档的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #初始化向量，类别1代表侮辱性文档，类别0代表正常文档
    p0Num = zeros(numWords); p1Num = zeros(numWords)
    #p0Denom和p1Denom分别代表该类别总词数。
    p0Denom = 0.0; p1Denom = 0.0
    for i in range(numTrainDocs):
        #遍历所有文件，如果是侮辱性文件，计算此侮辱性文件中出现的侮辱性单词的个数
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            #如果不是侮辱性文件，则计算非侮辱性文件中出现的非侮辱性单词的个数
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #类别1，既侮辱性文档[p(f1|c1),p(f2|c1),p(f3|c1)...]列表
    #在类别1的条件下，每个单词出现次数的概率
    p1Vect = p1Num/p1Denom #[1,2,3,4...]/50 --->[1/50,2/50,3/50,4/50...]
    #类别0,既正常文档[p(f1|c0),p(f2|c0),p(f3|c0)...]列表
    #在类别0的条件下，每个单词出现次数的概率
    p0Vect = p0Num/p0Denom #[1,2,3,4...]/50 -->[1/50,2/50,3/50,4/50...]
    return p0Vect,p1Vect,pAbusive
    
###############################################################################################

#代码4-3，朴素贝叶斯分类器训练函数优化版本
def trainNB0(trainMatrix,trainCategory):
    """
    Args:trainMatrix 文档矩阵，由一组组文档向量构成[1,0,0,1...]； trainCategory 每篇文档类别标签所构成的向量 [1,0,1,0,1,0]
    Return:p0Vect 正常文档中单词占比列表；p1Vect 侮辱性文档中单词占比列表；pAubsive 任意一篇文档属于侮辱性文档的概率。
    """
    #得到文档一共有多少个样本文件
    numTrainDocs = len(trainMatrix)
    #得到每组词向量的长度
    numWords = len(trainMatrix[0])
    #得到任意文档属于侮辱性文档的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #初始化向量，类别1代表侮辱性文档，类别0代表正常文档
    p0Num = ones(numWords); p1Num = ones(numWords)
    #p0Denom和p1Denom分别代表该类别总词数。
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        #遍历所有文件，如果是侮辱性文件，计算此侮辱性文件中出现的侮辱性单词的个数
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            #如果不是侮辱性文件，则计算非侮辱性文件中出现的非侮辱性单词的个数
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #类别1，既侮辱性文档[p(f1|c1),p(f2|c1),p(f3|c1)...]列表
    #在类别1的条件下，每个单词出现次数的概率
    p1Vect = log(p1Num/p1Denom) #[1,2,3,4...]/50 --->[1/50,2/50,3/50,4/50...]
    #类别0,既正常文档[p(f1|c0),p(f2|c0),p(f3|c0)...]列表
    #在类别0的条件下，每个单词出现次数的概率
    p0Vect = log(p0Num/p0Denom) #[1,2,3,4...]/50 -->[1/50,2/50,3/50,4/50...]
    return p0Vect,p1Vect,pAbusive

################################################################################################

#代码4-4，朴素贝叶斯分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    """
    Args:vec2Classify 指定一文档词向量[1,0,0...]; p0Vec 正常文档中单词占比列表；p1Vec 侮辱性文档中单词占比列表；pClass1 任意一篇文档属于侮辱性文档的概率。
    Return:该文档属于的类别，1 侮辱性文档，0 正常文档
    """
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
    
def testingNB():
    #创建数据集，返回listOPosts(单词列表)和listClasses(类别标签集合)
    listOPosts,listClasses = loadDataSet()
    #根据单词列表获得一个不重复的单词集合
    myVocabList = createVocabList(listOPosts)
    #创建一个空的列表
    trainMat = []
    #开始对listOPosts(单词列表)进行遍历
    for postinDoc in listOPosts:
        #把每一条文本转换为文档词向量[1,1,1,0...] 1代表出现，0代表为出现
        #添加到trainMat空列表中
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    #开始训练数据
    p0V,p1V,pAb = trainNB0(trainMat,listClasses)
    
    #测试样本
    testEntry = ['love','my','dalmation']
    #把单个测试样本转换为文档词向量[1,0,0....]
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    #利用朴素贝叶斯分类函数预测分类
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))


######################################################################################################

#代码4-5，朴素贝叶斯词袋模型
def bagOfWords2Vec(vocabList,inputSet):
    """
    Args: vocabList 所有单词集合列表，inputSet 输入文档
    Return: returnVec 返回文档词向量[1,2,3,4,2...]数字代表单词在文档中出现的次数。
    """
    returnVec = [0]*len(vocabList)
    #开始对输入的数据集进行遍历，出现加1。
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

#######################################################################################################

#代码4-6，文本解析及完整的垃圾邮件测试函数
def textParse(bigString):
    """文本解析
    """
    import re 
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    """垃圾邮件测试函数
    """
    docList = []; classList = []; fullText = []
    #开始循环文件夹内的所有文件
    for i in range(1,26):
        #对每封邮件文本解析，得到一个wordList.
        #垃圾文件
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        #添加到docList中.[[],[],[]]
        docList.append(wordList)
        #添加到所有文档中。
        fullText.extend(wordList)
        #在类别标签列表中添加1
        classList.append(1)
        
        #非垃圾邮件
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        #添加到docList中
        docList.append(wordList)
        #添加到所有文档中
        fullText.extend(wordList)
        #在类别标签列表中添加0
        classList.append(0)
    #得到所有的词列表
    vocabList = createVocabList(docList)
    
    #range(50)->[0,1,2,3,4...49]
    #创建一个训练集和测试集
    trainingSet = range(50); testSet = []
    #随机选择10个文档作为测试集，同时将其在训练集中剔除。
    for i in range(10):
        #random.uniform(x,y) 在x到y之间随机取值，包括x,不包括y.
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    #创建训练文档矩阵和训练类别标签集合
    trainMat = [] ; trainClasses = []
    #开始对训练集进行遍历[1,3,4,5,6,8....]
    for docIndex in trainingSet:
        #把每个文档转换为文档词向量，然后添加到训练文档矩阵中
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        #同时把对应的类别也添加到训练类别标签集合中
        trainClasses.append(classList[docIndex])
    #开始对训练集进行训练，并返回每个类别中单词占比列表
    p0V,p1V,pSpam = trainNB0(trainMat,trainClasses)
    
    #开始测试算法
    errorCount = 0
    #对随机构成的测试集进行遍历
    for docIndex in testSet:
        #同理，把测试集中的每个文档转换为文档词向量
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        #利用贝叶斯分类器对每个文档词向量进行分类，并且判断分类是否正确。
        if classifyNB(wordVector,p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    #最后输出总的错误百分比。
    print('the error rate is :',float(errorCount)/len(testSet))
    
#########################################################################################################

#代码4-7，RSS源分类器及高频词去除函数
def calcMostFreq(vocabList,fullText):
    """
    Args: vocabList 所有的单词列表；fullText 全部文档内容
    Return: 返回出现频率最高的前30个单词，{'hello': 100,'the':97...}
    """
    import operator
    freqDict = {}
    #开始对所有单词列表进行遍历
    for token in vocabList:
        #分别计算每个单词出现的次数
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    """
    """
    import feedparser
    #docList 文档列表，classList 类别标签列表。fullText 所有文档内容
    docList = []; classList = []; fullText = [];
    
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        #feed1和feed0做同理处理。
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    #生成全部单词列表
    vocabList = createVocabList(docList)
    #得到出现频率最高的前30个单词列表
    top30Words = calcMostFreq(vocabList,fullText)
    
    #开始对频率最高的30个单词列表循环，并在全部单词列表中剔除
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    #训练数据集和测试数据集
    trainingSet = range(2*minLen); testSet = [] 
    #随机选出20个样本组成测试数据集，并从训练样本中剔除.
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
        
    #生成训练文档矩阵和训练类别标签列表
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        #把每个样本转换成文档词向量.
        trainMat.append(bagOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    #训练数据集,生成各类别中单词出现的频率列表.
    p0V,p1V,pSpam = trainNB0(trainMat,trainClasses)
    #测试算法
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(wordVector,p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('teh error rate is :',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

#####################################################################################################
    
