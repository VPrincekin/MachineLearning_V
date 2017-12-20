#conding=utf-8
from numpy import *
import regression
"""
案例一：我们将回归用于真实数据
"""
if __name__ == '__main__':
    """#####################################################################################################################"""
    xArr,yArr = regression.loadDataSet(r'C:\Users\v_wangdehong\PycharmProjects\MachineLearning_V\Regression\data\abalone.txt')
    #使用前99行数据测试算法
    yHat01 = regression.lwlrTest(xArr[0:99],xArr[0:99],yArr[0:99],0.1)
    yHat1 = regression.lwlrTest(xArr[0:99],xArr[0:99],yArr[0:99],1)
    yHat10 = regression.lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 10)
    print(regression.rssError(yArr[0:99],yHat01))  #56.7842091184
    print(regression.rssError(yArr[0:99],yHat1))   #429.89056187
    print(regression.rssError(yArr[0:99],yHat10))  #549.118170883
    """
    从上面可以看到，使用较小的核将得到较低的误差，那么为什么不在所有数据集上都使用最小的核呢？
    因为使用最小的核将造成过拟合，对新数据不一定能达到最好的效果，下面就看看它在新数据上的表现
    """
    yHat01 = regression.lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], 0.1)
    yHat1 = regression.lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], 1)
    yHat10 = regression.lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], 10)
    print(regression.rssError(yArr[100:199], yHat01))  # 25119.4591112
    print(regression.rssError(yArr[100:199], yHat1))  # 573.52614419
    print(regression.rssError(yArr[100:199], yHat10))  # 517.571190538
    """
    从上面结果可以看到，核大小等于10时测试误差最小，但是它在训练集上的误差却是最大的。
    接下来再和简单的线性回归做个比较。
    """
    ws = regression.standRegres(xArr[0:99],yArr[0:99])
    yHat = mat(xArr[100:199])*ws #shape(99,1)
    print(regression.rssError(yArr[100:199],yHat.T.A))
    """
    简单的线性回归达到了局部加权线性回归类似的效果。这也表明了一点，必须在未知数据上比较效果才能选取到最佳模型。
    """



