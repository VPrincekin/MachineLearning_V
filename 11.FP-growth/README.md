### 使用FP-growth算法来高效发现频繁项集
在第10章中我们已经介绍了用Apriori 算法发现频繁项集与关联规则。本章将继续关注发现频繁项集这一任务，并使用FP-growth算法更有效的挖掘频繁项集。

FP-growth算法是一种用于发现数据集中频繁模式的有效方法，可以使用FP-growth算法在多种文本文档中查找频繁单词等。

### 代码思路顺序：

**treeNode ---> fpGrowth ---> excavate_fpGrowth ---> fpGrowth_Demo**

### 代码大致结构：

#### treeNode模块:
    定义构建FP树所需的数据结构，后续会从FP树种挖掘频繁项集。
    
    1. treeNode
        FP树的一个类，定义了FP的数据结构。
        
    2. main
        测试了构建FP树的数据结构。
#### fpGrowth模块：
    基于数据构建FP树
    
    1. createTree(dataSet, minSup=1)
        生成FP树
        -1.遍历所有的数据集合，计算所有项的支持度。
        -2.丢弃非频繁的项.
        -3.基于支持度降序排序所有的项。
        -4.所有数据集合按照得到的顺序重新整理。
        -5.重新整理完成后，丢弃每个集合末尾非频繁的项。 
        -6.读取每个集合插入FP树中，同时用一个头部链表数据结构维护不同集合的相同项。
    2. updateTree(items, inTree, headerTable, count)
        该函数的目的是为了让FP树生长。
        首先测试事务中的第一个元素项是否作为子节点存在。如果存在的话，则更新该元素项的计数。如果不存在，则创建一个新的treeNode并将其作为一个子节点添加到树中。
        这时，头指针表也要更新以指向新的节点。更新头指针表需要调用函数updataHeader().
        
    3. updateHeader(nodeToTest, targetNode)
        该函数用于更新头指针，确保节点连接指向树中该元素项的每一个实例。
        从头指针的nodeLink开始，一直沿着nodeLink直到到达链表末尾。这就是链表。
    
    4. loadSimpDat()
        自定义的数据集
        
    5. createInitSet(dataSet)
        简单的数据包装器
    
    6. main
        用自定义数据测试生成FP树。
        
#### excavate_fpGrowth模块：
    从FP树种挖掘出频繁项集（三个基本步骤）
        (1). 从FP树种获得条件模式基。（条件模式基是以所查找元素项为结尾的路径集合，每一条路径其实都是一条前缀路径。《左边路径，右边是值》）
        (2). 利用条件模式基，构建一个条件FP树。
        (3). 迭代重复步骤1，步骤2，直到树包含一个元素项为止。
    
    1. ascendTree(leafNode, prefixPath)
        迭代上溯整棵FP树，收集所有遇到的元素项的名称。
    
    2. findPrefixPath(basePat, treeNode)
        为给定元素生成一个条件模式基。
        遍历链表直到结尾，每遇到一个元素项都会调用ascendTree来上溯FP树。
        
    3. mineTree(inTree, headerTable, minSup, preFix, freqItemList)
        利用条件模式基递归查找频繁项集的minTree函数
        
    4. main
        测试给定元素项返回的条件模式基
        测试利用条件模式基递归查找频繁项集