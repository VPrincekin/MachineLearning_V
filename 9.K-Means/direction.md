### 利用K-Means聚类算法对未标注数据分组
聚类是一种无监督的学习, 它将相似的对象归到一个簇中, 将不相似对象归到不同簇中.相似这一概念取决于所选择的相似度计算方法.
K-Means 是发现给定数据集的 K 个簇的聚类算法, 之所以称之为 K-均值 是因为它可以发现 K 个不同的簇, 且每个簇的中心采用簇中所含值的均值计算而成。

### 代码思路顺序：
    
**kMeans ---> bisectiong_kMeans**

### 代码大致结构：

#### kMeans模块：
    K-Means聚类算法核心代码
   
    1. loadDataSet(fileName)
        加载数据函数。
        
    2. distEclud(vecA, vecB)
        计算两个向量的欧式距离函数(可根据场景不同自行选择)
    
    3. randCent(dataSet, k)
        该函数为给定数据集构建一个包含K个随机质心的集合。
        随机质心必须要在整个数据集的边界之内，这可以通过找到数据集每一维的最小和最大值来完成。
        然后生成 0~1.0 之间的随机数并通过取值范围和最小值，，以便确保随机点在数据的边界之内。
        
    4. kMeans(dataSet, k, distMeas=distEclud, createCent=randCent)
        完整的K-均值算法。该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。
        这个过程重复数次，直到数据点的簇分配结果不再改变为止。
    
    5. main
        测试randCent()函数,看看是否能生成min到max之间的值
        测试一下距离计算方法
        测试完整的K-均值算法
        
#### bisectiong_kMeans模块：
    二分K-Means聚类算法核心代码
    
    1. biKmeans(dataSet, k, distMeas=kMeans.distEclud)
        给定数据集、期望的簇数目和距离计算方法的条件下，返回聚类结果。
        
    2. main
        测试二分K-Means聚类算法