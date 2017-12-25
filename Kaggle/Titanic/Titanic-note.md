#### 比赛项目：

- 项目名称：泰坦尼克号: 灾难中的机器学习
- 项目地址：https://www.kaggle.com/c/titanic
- 项目级别：新手入门
	
#### 文档地址：
- GitHub: github.com/apachecn/kaggle
	
#### 参考资料：
- 如何在 Kaggle 首战中进入前 10%:	https://dnc1994.com/2016/04/rank-10-percent-in-first-kaggle-competition/
- Kaggle入门:	https://zhuanlan.zhihu.com/p/24883666?utm_source=qq&utm_medium=social
- Kaggle 泰坦尼克号解决方案:	https://www.kaggle.com/startupsci/titanic-data-science-solutions?scriptVersionId=1145136
- Pandas 十分钟入门:	https://www.cnblogs.com/chaosimple/p/4153083.html 
- Seaborn 官方教程:	http://seaborn.pydata.org/tutorial/aesthetics.html
	
#### 前置技能：
- numpy
- random
- matplotlib
- pandas:	pandas 官方文档: http://pandas.pydata.org/pandas-docs/stable/
			Pandas 十分钟入门: https://www.cnblogs.com/chaosimple/p/4153083.html 
- seaborn:	Seaborn 官方教程:http://seaborn.pydata.org/tutorial/aesthetics.html
	
- scikit-learn: 

            sklearn.svm --> SVC, LinearSVC
            sklearn.ensemble --> RandomForestClassifier
            sklearn.linear_model --> LogisticRegression
            sklearn.neighbors --> KNeighborsClassifier
            sklearn.naive_bayes --> GaussianNB
            sklearn.linear_model --> Perceptron
            sklearn.linear_model --> SGDClassifier
            sklearn.tree --> DecisionTreeClassifier
#### 开发环境：
- 平台：Windows
- python: 集成环境: Anaconda3 (64-bit)
- IDE: PyCharm + Jupyter Notebook


#### 概述：
- **比赛说明**：RMS泰坦尼克号的沉没是历史上最臭名昭着的沉船之一。1912年4月15日，在首航期间，泰坦尼克号撞上一座冰山后沉没，2224名乘客和机组人员中有1502人遇难。
这一耸人听闻的悲剧震撼了国际社会，并导致了更好的船舶安全条例。沉船导致生命损失的原因之一是乘客和船员没有足够的救生艇。虽然幸存下来的运气有一些因素，但一些人比其他人更有可能生存，比如妇女，儿童和上层阶级。
在这个挑战中，我们要求你完成对什么样的人可能生存的分析。
特别是，我们要求你运用机器学习的工具来预测哪些乘客幸存下来的悲剧。
- **目标**：你的工作是去预测沉默的泰坦尼克号是否有幸存下来的乘客。
针对测试机中的每个 PassengerID，对于 Survived 变量您必须预测一个 0 或 1 的值。
- **度量**：您的分数是您正确预测的乘客的百分比。这被称为 "准确性"。
- **提交文件的格式**：你应该提交一个csv文件，正好有418个条目和一个标题行。如果您有额外的列（超出PassengerId和Survived）或行，您的提交将会显示错误。
#### 数据：
- **数据集**：
    1. 训练集（train.csv）   
    2. 测试集（test.csv）  
    3.提交样本示例（gender_submission.csv）。
- **数据字典**：
    
    
    1.survival(是否生存)：0=否，1=是。
    2.pclass(船票类别)：1=1st,=2nd,3=3rd。
    3.sex(性别)：male=男，famale=女
    4.Age(年龄)
    5.slibsp(泰坦尼克号上的兄弟姐妹/配偶）
    6.parch(泰坦尼克号上的父母/孩子)
    7.ticket(船票号码)
    8.fare(旅客票价)
    9.cabin(房间号)
    10.embarked(出发港): C=Cherbourg,Q=Queenstown,S=Southampton
    
#### 教程：

- **工作流程阶段**:
      
        
    1.问题或问题的定义。
    2.获取 training（训练）和 testing（测试）数据.
    3.Wrangle（整理）, prepare（准备）, cleanse（清洗）数据
    4.Analyze（分析）, identify patterns 以及探索数据.
    5.Model（模型）, predict（预测）以及解决问题.
    6.Visualize（可视化）, report（报告）和提出解决问题的步骤以及最终解决方案.
    7.提供或提交结果.
    
- **工作流程目标：数据科学解决方案工作流程有以下七个主要的目标.**

    
    1.Classifying（分类）
        我们可能想对我们的样本进行分类或加以类别. 我们也可能想要了解不同类别与解决方案目标的含义或相关性.
    2.Correlating（相关）
        可以根据训练数据集中的可用特征来处理这个问题. 数据集中的哪些特征对我们的解决方案目标有重大贡献？从统计学上讲, 特征和解决方案的目标中有一个相关。
        随着特征值的改变, 解决方案的状态也会随之改变, 反之亦然。这可以针对给定数据集中的数字和分类特征进行测试.
        我们也可能想要确定以后的目标和工作流程阶段的生存以外的特征之间的相关性. 关联某些特征可能有助于创建, 完善或纠正特征。
    3.Converting（转换）
        对于建模阶段, 需要准备数据. 根据模型算法的选择, 可能需要将所有特征转换为数值等价值. 所以例如将文本分类值转换为数字的值.
    4.Completing（完整）
        数据准备也可能要求我们估计一个特征中的任何缺失值. 当没有缺失值时，模型算法可能效果最好.
    5.Correcting（校正）
        我们还可以分析给定的训练数据集以找出错误或者可能在特征内不准确的值, 并尝试对这些值进行校正或排除包含错误的样本. 
        一种方法是检测样本或特征中的任何异常值. 如果对分析没有贡献, 或者可能会显着扭曲结果, 我们也可能完全丢弃一个特征.
    6.Creating（创建）
        我们可以根据现有特征或一组特征来创建新特征, 以便新特征遵循 correlation（相关）, conversion（转换）, completeness（完整）的目标.
    7.Charting（绘图）
        如何根据数据的性质和解决方案的目标来选择正确的可视化图表工具以及绘图.
        
- **导入需要的包：**


    # data analysis and wrangling
    import pandas as pd
    import numpy as np
    import random as rnd

    # visualization
    import seaborn as sns
    import matplotlib.pyplot as plt
    %matplotlib inline

    # machine learning
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC, LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    
- **获取数据**：
    
        
    描述: Python 的 Pandas 包帮助我们处理我们的数据集. 
    我们首先将训练和测试数据集收集到 Pandas DataFrame 中.
    我们还将这些数据集组合在一起, 在两个数据集上运行某些操作.
    
    train_df = pd.read_csv('../input/train.csv')
    test_df = pd.read_csv('../input/test.csv')
    combine = [train_df, test_df]
    
- **分析数据：**

#####数据集中哪些特征可用？
   
   
    print(train_df.columns.values)

#####哪些特征是分类的？

    train_df.head()
    
#####哪些特征是数值的？
    
    train_df.head()
    
#####哪些特征是混合的数据类型？
    
    train_df.tail()

#####哪些特征可能包含错误或错别字？
    
    train_df.tail()   

#####哪些特征包含 blank，null 或 空值？

    train_df.info()
    print('_'*40)
    test_df.info()
    
#####各个特征的数据类型是什么样的？

    train_df.info()
    print('_'*40)
    test_df.info()
    
######样本中数值的特征值分布是什么样的？
    
    1.总样本是 891 或者在泰坦尼克号（2,224）上实际旅客的 40%.
    2.Survived（生存）是一个具有 0 或 1 值的分类特征.
    3.大约 38% 样本幸存了下来, 然而实际的幸存率是 32%.
    4.大多数旅客 (> 75%) 没有和父母或孩子一起旅行.
    5.近 30% 的旅客有兄弟姐妹 和/或 配偶.
    6.少数旅客 Fares（票价）差异显著 (<1%), 最高达 $512.
    7.很少有年长的旅客 (<1%) 在年龄范围 65-80.
    
#####分类特征的分布是什么样的?  

    1.Names（名称）特征在数据集中是唯一的 (count=unique=891)
    2.Sex（性别）变量有两个可能的值, 男性为 65% (top=male, freq=577/count=891).
    3.Cabin（房间号）值在样本中有重复. 或者几个旅客共享一个客舱.
    4.Embarked（出发港）有 3 个可能的值. 大多数乘客使用 S 港口(top=S)
    5.Ticket（船票号码）特征有很高 (22%) 的重复值 (unique=681).
    
#####基于数据分析的假设：
    
    1.Correlating（相关）
        我们想知道每个特征与生存相关的程度. 我们希望在项目早期做到这一点, 并将这些快速相关性与项目后期的模型相关性相匹配.
    2.Completing（完整）
        我们可能想要去补全丢失的 Age（年龄）特征，因为它肯定与生存相关.
        我们也想要去补全丢失的 Embarked（出发港）特征, 因为它也可能与生存或者其它重要的特征相关联.
    3.Correcting（校正）
        Ticket（船票号码）特征可能会从我们的分析中删除, 因为它包含了很高的重复比例 (22%), 并且票号和生存之间可能没有关联.
        Cabin（房间号）特征可能因为高度不完整而丢失, 或者在 训练和测试数据集中都包含许多 null 值.
        PassengerId（旅客ID）可能会从训练数据集中删除, 因为它对生存来说没有贡献.
        Name（名称）特征是比较不规范的, 可能不直接影响生产, 所以也许会删除.
    4.Creating（创建）
        我们可能希望创建一个名为 Family 的基于 Parch 和 SibSp 的新特征，以获取船上家庭成员的总数.
        我们可能想要设计 Name 功能以将 Title 抽取为新特征.
        我们可能要为 Age（年龄）段创建新的特征. 这将一个连续的数字特征转变为一个顺序的分类特征.
        如果它有助于我们的分析, 我们也可能想要创建 Fare（票价）范围的特征。
    5.Classifying（分类
        根据前面提到的问题描述, 我们也可以增加我们的假设. 
            Women (Sex=female) 更有可能幸存下来.
            Children (Age<?) 更有可能幸存下来.
            上层阶级的旅客 (Pclass=1) 更有可能幸存下来.

- **通过旋转特征进行分析**


    为了确认我们的一些观察和假设, 我们可以快速分析我们的特征之间的相互关系.
    我们只能在这个阶段为没有任何空值的特征做到这一点.
    对于 Sex（性别），顺序的（Pclass）或离散的（SibSp，Parch）类型的特征, 这也是有意义的.
    
- **通过可视化数据进行分析**

    观察 --->决策
    
- **整理数据：**
    
    **1.删除特征以校正数据**
        这是一个很好的开始执行目标. 通过丢弃特征, 我们正在处理更少的数据点. 加快我们的 notebook, 并简化分析.
        根据我们的假设和决策, 我们要放弃 Cabin（房间号）（更正＃2）和 Ticket（票号）（更正＃1）的特征.
        请注意, 如果适用, 我们将对训练和测试数据集进行操作, 以保持一致.
    
    **2.从现有的特征中创建新特征**
        我们可以用更常见的头衔来替换很多头衔 或者将它们分类为 `Rare`.
        我们可以将 titles（头衔）转换为顺序的.
        现在我们可以放心地从训练和测试数据集中删除 Name 特征. 我们也不需要训练数据集中的 PassengerId 特征.
    **3.转换分类的特征**
        现在我们可以将包含字符串的特征转换为数字值.这是大多数模型算法所要求的.这样做也将帮助我们实现特征完成目标.
        让我们开始将 Sex（性别）特征转换为名为 Gender（性别）的新特征, 其中 female=1, male=0.
    **4. 完整化数值字连续特征**
        现在我们应该开始估计和完成缺少或空值的特征. 我们将首先为 Age（年龄）特征执行此操作.我们可以考虑三种方法来完整化一个数值连续的特征.
        更准确地猜测缺失值的方法是使用其他相关特征.
        在我们的例子中, 我们注意到 Age（年龄）, Sex（性别）和 Pclass 之间的相关性.
        猜测年龄值使用 中位数 Age 中的各种 Pclass 和 Gender 特征组合的值.
        因此, Pclass=1 和 Gender=0，Pclass=1 和 Gender=1 的年龄中位数等等...
    **5. 结合现有特征创建新特征**
        我们可以为 Parch 和 SibSp 结合的 FamilySize 创建一个新的特征.这将使我们能够从我们的数据集中删除 Parch 和 SibSp.  
    **6. 完整化分类特征**
        Embarked（出发港）特征有 S, Q, C 三个基于出发港口的值. 我们的训练集有两个丢失值. 我们简单的使用最常发生的情况来填充它.
    **7. 转换分类特征为数值的**
        我们现在可以通过创建一个新的数字港特征来转换 EmbarkedFill 特征.（S--0.C---1.Q--2）
     
- **模型预测和解决问题**
    
    
        Logistic Regression
        KNN or k-Nearest Neighbors
        Support Vector Machines
        Naive Bayes classifier
        Decision Tree
        Random Forrest
        Perceptron
        Artificial neural network
        RVM or Relevance Vector Machine

- **参考文献**
    泰坦尼克号之旅：https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic   
    Pandas 入门指南: Kaggle 的泰坦尼克号竞赛： https://www.kaggle.com/c/titanic/details/getting-started-with-random-forests
    泰坦尼克号的最佳处理分类器：https://www.kaggle.com/sinakhorami/titanic/titanic-best-working-classifier