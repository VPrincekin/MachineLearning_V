========================================================================================================
#### 数据结构
1. 列表； list ['a','b','c']
   list是一个可变的有序表。list里面的元素数据类型也可以不同。
   列表方法：
            del list(index)  删除元素  
            
            append 在列表末尾追加新的对象
            extend 在列表末尾一次性追加另一个列表的多个值
            count 计算某个元素在列表中出现的次数
            pop 移除列表中的一个元素（默认是最后一个），并且返回该元素的值。
            reverse 将列表中的元素方向存放
            intersection 判断一个列表中是否有某个元素
            
            对itemScores二维列表按照第二个元素排序，True代表降序。[:N]返回前N个
                itemScores=[['a',1],['b',2],['c',3]]
            sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

2. 元组：tuple 元组与列表一样，也是一种序列。唯一的不同是元组不能修改。

3. 字典：Map {key:value,key:value,...}
        update()函数：
            a.update(b)
            如果字典b中的a中没有，就追加元素，如果有，就更新元素。

4. 数组：Numpy模块  array(list)
        
5. 矩阵：Numpy模块  mat(list)

==========================================================================================================
#### Numpy模块

1. random.uniform(n,m)  返回一个n~m之间的随机数。  
2. mat 把传入的一个列表转换为numpy矩阵，为了方便计算。
3. .transpose和.T一样的功能，矩阵的转置。  
4. shape    查看矩阵的大小，返回（m,n）。   
5. zeros((m,n))     生成一个m x n 都值为0的矩阵。  
6. multiply(A,B)    A,B两个矩阵对应元素相乘。
7. svInd = nonzero(alphas.A > 0)[0]  得到alphas矩阵中所有大于0元素的位置
8. arange(m)    可以生成一个从0到m-1的数组。
9. random.shuffle(indexList)  numpy提供的shuffle函数可以对indexList中的元素进行混洗。
10. 对于一个矩阵a:
    a.dtype     查看矩阵的数据类型
    a.astype('float64')    改变矩阵的数据类型
11. 矩阵去重
    import numpy as np
    a = np.array([[1,2,3],[1,2,3],[1,2,5]])
    print(np.array(list(set(tuple(t) for t in a))))
=========================================================================================================
#### IO
1. 我们从网络或磁盘上读取了字节流，那么读到的数据就是bytes。要把bytes变为str，就需要用decode()方法：
    bytes中只有一小部分无效的字节，可以传入errors='ignore'忽略错误的字节.

    b'\xe4\xb8\xad\xff'.decode('utf-8', errors='ignore')
    
2. input()返回的数据类型是str.

3.  **UnicodeEncodeError: 'gbk' codec can't encode character u'\u3232' in position 0: illegal multibyte sequence**
    解决方法：指定解码格式，忽略非法字符。  
    fr = open(inFile,encoding='utf-8',errors='ignore')

==========================================================================================================
### 数据特标准化：所有的特征都减去各自的均值并除以方差
    ```
    xMat = mat(xArr)
    yMat=mat(yArr).T
    计算Y的均值
    yMean = mean(yMat,0)
    #Y的所有特征减去均值
    yMat = yMat - yMean
    #计算X的均值(按照列计算)
    xMeans = mean(xMat,0)
    #计算X的方差
    xVar = var(xMat,0)
    #所有特征都减去各自的均值并除以方差
    xMat = (xMat - xMeans)/xVar
    ```
=======================================================================================================================