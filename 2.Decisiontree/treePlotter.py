#coding=utf-8

#代码3-6，使用文本注解绘制树节点。
import matplotlib.pyplot as plt
#定义文本框和箭头格式【sawtooth 波浪方框 round4 矩形方框 ，fc表示字体颜色深浅 0.1-0.9 依次变浅！ 】
decisionNode = dict(boxstyle='sawtooth',fc='0.8')
leafNode = dict(boxstyle='round4',fc='0.8')
arrow_args = dict(arrowstyle = "<-")
#绘制带剪头的注解
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords="axes fraction",xytext=centerPt,textcoords='axes fraction',\
                            va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)

#创建树图--测试版                         
"""
def createPlot():
    #创建一个figure的模版
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1=plt.subplot(111,frameon=False)
    plotNode('decision node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()
"""    
    
    
    
#代码3-7，获取叶节点的数目和树的层数
"""获取树的节点数目
Args:myTree 得到的树
Return:树的节点数目
"""
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        #判断测试节点的数据是否为字典
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else: numLeafs +=1
    return numLeafs

"""获取树的层数
Args:myTree 得到的树
Return:树的层数
"""
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1+getTreeDepth(secondDict[key])
        else: thisDepth = 1
        if thisDepth > maxDepth: 
            maxDepth = thisDepth
    return maxDepth



#代码3-8,编辑plotTree函数，并更新createPlot()函数
#在父子节点之间填充文本信息
def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)
#计算宽与高
def plotTree(myTree,parentPt,nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    #标记子节点属性值
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    #减少y偏移
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else :
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff, plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1,facecolor = 'green')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree,(0.5,1.0),'')
    plt.show() 
    