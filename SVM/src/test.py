#导入模块
import svmMLiA
#加载数据
dataArr,labelArr=svmMLiA.loadDataSet('C:/Users/v_wangdehong/PycharmProjects/MachineLearning_V/SVM/digits/testSet.txt')

#训练算法，得到alphas和b
#b,alphas=svmMLiA.smoSimple(dataArr,labelArr,0.6,0.01,40)

b,alphas = svmMLiA.smoP(dataArr,labelArr,0.6,0.001,40)

#print(b,alphas)

#通过alphas计算w

ws = svmMLiA.calcws(alphas,dataArr,labelArr)
print(ws)