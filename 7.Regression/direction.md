### 代码思路顺序:

**regression--->regressionDemo1--->ridge_regression--->lasso_regression--->regressionDemo2**

### 代码大致结构：

#### regression模块：  
    
    1. loadDataSet(fileName) 
        加载文件数据函数
    2. standRegres(xArr,yArr)
        标准回归函数，用来计算回归系数w
    3. lwlr(testPoint,xArr,yArr,k=1.0)
        局部加权线性回归函数，在标准回归函数上的优化，在待测点附近每个点赋予一定的权重，在子集上基于最小均方差来进行普通的回归。
        主要就是对w进行处理：计算每个样本点的权重值，随着样本点与待测点距离的递增，权重将以指数级衰减。
    4. lwlrTest(testArr,xArr,yArr,k=1.0)
        测试局部加权线性回归，上面的函数是单点的，这个是对数据集中每个点调用lwlr()函数。
        这里的k是关于赋予权重矩阵的核函数的一个参数，与权重的衰减速率有关，可以人为调控。
    5. rssError(yArr,yHatArr)
        计算测试结果误差的函数，计算的是所有测试数据点的总和
     
#### regressionDemo1模块：
     
    用regression模块中的函数来预测鲍鱼的年龄案例。
   
#### ridge_ression模块：
   
    1. ridgeRegres(xMat,yMat,lam=0.2)
        通过岭回归计算公式得到回归系数w.
        当我们的数据特征比样本点还多时，再用之前的线性回归就会出问题，这是我们需要使用岭回归函数。
        简单来说，岭回归就是在原有的矩阵上加一个 λI 从而使得矩阵非奇异。
    2. ridgeTest(xArr,yArr)
        在一组 λ 上测试结果，返回的是在不同 λ 情况下对应的回归系数。
        注意在这个过程中需要对数据做标准化处理；x：所有特征都减去各自的均值并除以方差。y：所有特征减去均值。
    3. main
        通过绘图看出回归系数的缩减效果。
        缩减方法可以去掉不重要的参数，因此能更好地理解数据。
        
#### lasso_regression模块：
   
    1. regularize(xMat)
        数据特征标准化
    2. stageWise(xArr,yArr,eps=0.01,numIt=100)
        前向逐步线性回归算法。返回的是每次迭代结束对应的回归系数w所组成的矩阵。
        一种贪心算法，即每一步都尽可能减少误差。一开始，所有权重都设置为 1，然后每一步所做的决策是对某个权重增加或减少一个很小的值。
        逐步线性回归算法的主要优点在于它可以帮助人们理解现有的模型并作出改进。
        当构建了一个模型后，可以运行该算法找出重要的特征，这样就有可能及时停止对那些不重要特征的收集。
    3. main
        通过绘图看出回归系数在每次迭代过程中的缩减效果。
        
#### regressionDemo2模块：
   
    1. scrapePage(retX, retY, inFile, yr, numPce, origPrc)
        从页面读取数据，生成retX和retY列表。
    2. setDataCollect(retX, retY)
        依次读取六种乐高套装的数据，并生成数据矩阵
    3. crossValidation(xArr, yArr, numVal=10)
        交叉验证测试岭回归函数
    4. main
        测试效果
        我们可以看一下在缩减过程中回归系数是如何变化的，我们可以通过不同的变化来分析哪些特征是关键的，哪些特征是不重要的。
        