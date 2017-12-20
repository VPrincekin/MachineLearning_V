### 树回归

我们在Regression中介绍了线性回归的一些强大的方法，但这些方法创建的模型需要拟合所有的样本点（局部加权线性回归除外）。
当数据拥有众多特征并且特征之间关系十分复杂时，构建全局模型的想法就显得太难了，也略显笨拙。而且，实际生活中很多问题都是非线性的，不可能使用全局线性模型来拟合任何数据。
一种可行的方法是将数据集切分成很多份易建模的数据，然后利用我们的线性回归技术来建模。如果首次切分后仍然难以拟合线性模型就继续切分。在这种切分方式下，树回归和回归法就相当有用。
除了我们在Decisiontree中介绍的决策树算法，我们介绍一个新的叫做CART(Classification And Regression Trees,分类回归树)的树构建算法。该算法既可以用于分类还可以用于回归。

### 代码思路顺序:

**treeNode ---> regTrees ---> pruning_Tress ---> model_Tress ---> Regression_VS_RegTress**

### 代码大致结构:

#### treeNode模块:
    简单介绍了CART算法，介绍了树的数据结构。(本模块代码对后面不影响，仅起到说明作用)
 
#### regTrees模块：
    树回归的核心代码
    
    1. loadDataSet(fileName)
        解析文件数据函数
        
    2. binSplitDataSet(dataSet, feature, value)
        通过数组过滤方式将数据集合切分得到两个子集并返回。大于指定value的放到一边，小于的放到另一边
        
    3. regLeaf(dataSet)
        负责生产叶节点，当确定不再对数据进行切分时，将调用该函数来得到叶节点的模型。在回归树种，该模型其实就是目标变量的均值。
        
    4. regErr(dataSet)
        误差估计函数，计算总方差=方差*样本数
        
    5. chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4))
        回归树构建的核心函数，该函数的目的是找到数据的最佳二元切分方式。
        如果找不到一个‘好’的二元切分，该函数返回None并同时调用createTree()方法来产生叶节点，叶节点的值也将返回None.
    
    6. createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4))、
        树构建函数，一个递归函数。
        该函数首先尝试将数据集分成两个部分，切分由函数chooseBestSplit()完成。
        如果满足停止条件，将返回None和某类模型的值。如果构建的是回归树，该模型就是一个常数。如果是模型树，其模型就是一个线性方程。
        如果不满足停止条件，chooseBestSplit()会创建一个新的Python字典并将数据集分成两份，在这两份数据集上将分别继续递归调用createTree()函数。
    7. main
        测试binSplitDataSe函数.
        测试完整的createTree()函数
        多次切分的例子
        
#### pruning_Tress模块：
    对树的剪枝核心代码，采用的是后剪枝。
    
    1. isTree(obj)
        判断输入变量是否是一棵树，返回布尔类型的结果。
        
    2. getMean(tree)
         一个递归函数，它从上往下遍历树直到叶节点为止。如果找到两个叶节点则计算他们的平均值。
        该函数对树进行塌陷处理，即返回树的平均值。
        
    3. prune(tree, testData)
        该函数首先需要确认测试集是否为空。一旦非空，则反复递归调用函数prune()对测试数据进行切分。
        接下来要检查某个分支到底是子树还是节点。如果是子树，就调用函数prune()来对该子树进行剪枝。
        在对左右两个分支完成剪枝之后，还需要检查他们是否仍然还是子树，如果两个分支已经不再是子树，那么就可以进行合并。
        具体的做法是对合并前后的误差进行比较。如果合并后的误差比不合并的误差小就进行合并，否则的话不合并直接返回。
     
    4. main
        开始测试剪枝效果

#### model_Tress模块：
    构建模型树的核心代码，模型树的可解释性是它优于回归树的特点之一。另外，模型树也具有更高的预测准确度。
    
    1. linearSolve(dataSet)
        该函数主要是将数据集格式化成目标变量Y和自变量X。进行标准线性回归计算，可以得到回归系数ws.
     
    2. modelLeaf(dataSet)
        该函数的功能与regTrees模块中的regLeaf()类似，负责生产叶节点，当确定不再对数据进行切分时，将调用该函数来得到叶节点的模型。
        不同的是，该函数在数据集上调用linearSolve()并返回回归系数ws
        
    3. modelErr(dataSet)
        该函数可以在给定的数据集上计算误差。与regTrees模块中的regErr()类似，会被chooseBestSplit()调用来找到最佳切分。
        不同的是，该函数在数据集上调用linearSolve()并返回回归系数ws，X，Y。最后返回预测值和实际值之间的平方误差
        
    4. main 
        测试模型树实际效果
        
#### Regression_VS_RegTress模块：
    前面介绍了模型树、回归树和一般的回归方法，下面测试一下哪个模型最好。
    
    1. regTreeEval(model, inDat)
        回归树测试案例
        
    2. modelTreeEval(model, inDat)
        模型树预测案例
        
    3. treeForeCast(tree, inData, modelEval=regTreeEval)
        计算预测的结果
        在给定树结构的情况下，对于单个数据点，该函数会给出一个预测值。
        该函数自顶向下遍历整棵树，直到命中叶节点为止，一旦到达叶节点，它就会在输入数据上调用modelEval()函数，该函数的默认值为regTreeEval()
        
    4. createForeCast(tree, testData, modelEval=regTreeEval)
         多次调用treeForeCast()函数，对特定模型的树进行预测，可以是 回归树 也可以是 模型树。
         
    5. main 
         创建一棵回归树测试
         创建一棵模型树测试
         使用标准线性回归测试