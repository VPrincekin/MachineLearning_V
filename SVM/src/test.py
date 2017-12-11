#导入模块
import svmMLiA
import svmDemo
import svm
#加载数据
# dataArr,labelArr=svmMLiA.loadDataSet('C:/Users/v_wangdehong/PycharmProjects/MachineLearning_V/SVM/digits/testSet.txt')
# dataArr, labelArr = svmDemo.loadImages('C:/Users/v_wangdehong/PycharmProjects/MachineLearning_V/SVM/digits/trainingDigits')
# print(dataArr,labelArr)
#训练算法，得到alphas和b
# b,alphas=svmMLiA.smoP(dataArr, labelArr, 200, 0.0001, 10000,('rbf',20))

#b,alphas = svmMLiA.smoP(dataArr,labelArr,0.6,0.001,40)

# print(b,alphas[alphas>0])

#通过alphas计算w

#ws = svmMLiA.calcws(alphas,dataArr,labelArr)

svmDemo.testDigits(('rbf',10))
# svm.testDigits(('rbf',10))？