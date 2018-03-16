### 我该如何开始一个机器学习项目？

### 一、数据的加载

- 首先导入所需要的包,我们一般使用 pandas 处理分析数据
```python
import numpy as np 
import pandas as pd
```
- 从csv文件中加载数据
```python
import pandas as pd
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
```
- 从文本文件中加载数据
```python
def loadDataSet(fileName):
    """
    加载文件数据
    :param fileName:
    :return:
    """
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
```

### 二、数据的探索
通过对数据进行探索，我们可以对数据的基本特征有一个全局的大致了解。

- 查看数据的基本布局信息
```python
train.head() #可以查看(默认)前5行数据信息。
train.tail() #可以查看后10行数据信息。
```
- 查看数据都有哪些特征
```python
train.columns
```
- 查看数据的基本情况
```python
train.describe()
```
- 当然也可以查看某一指定特征的具体情况
```python
train["feature"].describe()
```
- 利用直方图查看某一特征数据的具体分布情况(常用来查看目标变量是否符合正态分布)
```python
import seaborn as sns
sns.distplot(train["feature"])
```
- 利用散点图分析变量间的关系(常用来发现某些离散点)
```python
import matplotlib.pyplot as plt
output,var,var1,var2 = 'SalePrice','GrLivArea','TotalBsmtSF','OverallQual'
fig,axis = plt.subplots(1,3,figsize=(16,5))
train.plot.scatter(x=var,y=output,ylim=(MIN_VALUE,MAX_VALUE),ax=axis[0])
train.plot.scatter(x=var1,y=output,ylim=(MIN_VALUE,MAX_VALUE),ax=axis[1])
train.plot.scatter(x=var2,y=output,ylim=(MIN_VALUE,MAX_VALUE),ax=axis[2])
```
- 利用 seaborn 对多个特征的散点图、直方图进行整合，得到各个特征两两组合形成的图矩阵(用来查看特征直接的相关性)
```python
import matplotlib.pyplot as plt
import seaborn as sns
var_set = ["feature1","feature2","feature3","feature4","feature5","feature6"]
sns.set(font_scale=1.25) #设置坐标轴的字体大小
sns.pairplot(train[var_set]) ## 可在kind和diag_kind参数下设置不同的显示类型，此处分别为散点图和直方图，还可以设置每个图内的不同类型的显示
plt.show()
```
- 利用热力图对各个特征间的关系进行分析
```python
import matplotlib.pyplot as plt
import seaborn as sns
corrmat = train.corr()
f,axis = plt.subplot(figsize=(14,12))
sns.heatmap(corrmat,vmax=0.8,square=True,ax=axis)
plt.show()
```
- 选取与目标变量相关系数最高的K个特征，找出那些相互关联性较强的特征
```python
import matplotlib.pyplot as plt
import seaborn as sns
corrmat = train.corr()
k = 10
top10_attr = corrmat.nlargest(k,output).index #output代表的是目标变量
top10_mat = corrmat.loc[top10_attr,top10_attr]
fig,axis = plt.subplots(figsize=(14,10))
sns.set(font_scale=1.25)
sns.heatmap(top10_mat,annot=True,annot_kws={"size":12},square=True)
plt.show()
```

- 通过一元方差分析，获得各个离散型变量对目标变量的影响
```python
def anova(data,columns):
    anv = pd.DataFrame()
    anv["feature"] = columns
    pvals = []
    for c in columns:
        samples = []
        for cls in data[c].unique():
            s = data[data[c]==cls]['SalePrice'].values
            samples.append(s) #某特征下不同取值对应的房价组合形成的二维列表
        pval = stats.f_oneway(*samples)[1] #一元方差分析得到 F，P，要的是 P，P越小，对方差的影响越大。
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

a = anova(train,category_feature)
a['disparity'] = np.log(1./a['pval'].values)
fig,axis = plt.subplots(figsize=(14,12))
sns.barplot(data = a ,x = 'feature',y='disparity')
#选择x轴变量名的位置
x = plt.xticks(rotation=90)
plt.show()
```

### 三、数据的预处理

对数据的预处理主要分为以下几个方面：
1. 缺失值的填补。
2. 把类别型特征转化为数值型特征。
3. 把数值型特征转化为类别型特征。
4. 删除离散点样本。
5. 删除无关性特征。
6. 添加新的特征。

#### 1.缺失值的填补
缺失值处理有两种方案：
一种是分析含缺失值的特征对任务有没有用，没有的特征直接删除，有用的特征依据缺失量，少则删除样本，多则用mean，median，mode补全。
另一种是分析这些缺失值缺失的原因，并用一定方法将其转换为一类数据(成为类型变量的一个类型)。

- 缺失值情况统计(先要看一下缺失值具体出现在那些特征中，缺失信息是否对整个特征有较大影响，缺失信息是否有其他特殊意义？)
```python
na_count = train.isnull().sum().sort_values(ascending = False)
na_count = na_count[na_count.values>0]
na_rate = na_count/len(train)
na_data = pd.concat([na_count,na_rate],axis = 1,keys = ["count",'ratio'])
```
- 处理方案一(删除多余或者不相关特征)
接下来我们就可以根据统计结果来对不同的特征缺失值做不同的处理。一般的，如果某一特征的数据缺失量达到15%以上，我们应该删除这些特征(当然具体还要看特征对目标变量的影响)。
对于缺失量很小的特征，我们可以直接删除缺失值的那些样本即可。还有一些特征可能代表的是某一种相同的信息，或者和已存在的某些特征具有较强的相关性，这样的话我们可以删除此类特征，保留一个有效特征即可。

```
train.drop()
```
- 处理方案二(补全与变换)
```python
#类型变量特征集合
category_feature = [attr for attr in train.columns if train.columns[attr]=='object']
#数值变量特征集合
number_feature = [attr for attr in train.columns if train.columns[attr]!='object']
```
类型变量特征缺失值补全(对于类型变量特征的缺失值，一般用样本中最多的，或者 Missing 填充。具情况而定)
```python
#使用 Missing 填充
train[feature].fillna("Missing",inplace = True)
#使用样本中最多的填充
train[feature].fillna(train[feature].mode()[0],inplace = True)
```
数值变量特征缺失值补全(对于数值型变量特征我们一般使用 mean , median, mode或者0。具情况而定)
```python
train[feature].fillna(0.,inplace = True)
train[feature].fillna(train[feature].mean(),inplace = True) #平均值
train[feature].fillna(train[feature].median(),inplace = True) #中位数
train[feature].fillna(train[feature].mode()[0],inplace = True) #众数
```
#### 2.把类别型特征转化为数值型特征
一般的我们可以使用 One-Hot编码来处理类别型特征。
```python
train = pd.get_dummies(train)

# df['B'] = df.replace({'a':0,'b':1,'c':2})  自定义替换
```
但在我们的特征中，有些类别特征明显与目标变量有较强的线性关系，对于此类特征，我们应将其转化为数值特征。
所有类型变量，依照各个类型变量的不同取值对应的样本集内房价的均值，按照房价均值高低对此变量的当前取值确定其相对数值1,2,3,4等等，相当于对类型变量赋值使其成为连续变量。此方法采用了与One-Hot编码不同的方法来处理离散数据，值得学习!
```python
def encode(data,feature,test):
    ordering = pd.DataFrame()
    ordering['val'] = data[feature].unique()
    ordering.index = ordering.val
    # groupby()操作可以将某一feature下同一取值的数据整个到一起，结合mean()可以直接得到该特征不同取值的房价均值
    ordering['price_mean'] = data[[feature,'SalePrice']].groupby(feature).mean()
    ordering = ordering.sort_values('price_mean')
    ordering['order'] = range(1,ordering.shape[0]+1)
    ordering = ordering['order'].to_dict()
    for attr, score in ordering.items():
        data.loc[data[feature] == attr, feature+'_E'] = score
        test.loc[test[feature] == attr, feature+'_E'] = score
        
for feature in category_feature:
    encode(train,feature,test)
train.drop(category_feature,axis=1,inplace=True)
test.drop(category_feature,axis=1,inplace=True)
```
- 应用斯皮尔曼等级相关系数，分析连续性变量与目标变量的相关性

```python
def spearman(train, features):
    '''
    采用“斯皮尔曼等级相关”来计算变量与房价的相关性(可查阅百科)
    此相关系数简单来说，可以对encoder()处理后的等级变量及其它与房价的相关性进行更好的评价（特别是对于非线性关系）
    '''
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['corr'] = [train[f].corr(train['SalePrice'], 'spearman') for f in features]
    spr = spr.sort_values('corr')
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=spr, y='feature', x='corr', orient='h')    
    
features = train.columns
spearman(train, features)
```

- 我们还可以利用热力图，分析数值型变量和类别型变量之间的相关性。离散变量间、连续变量间，两种变量间某些变量存在很强的互相关型，在特征选取时，应在这些互相关的特征中n选1

```python
features = train.columns #此时的 train 是已经把类别特征转换成数值特征的数据集

category_E= []
for q in category_feature:
    category_E.append(q+"_E")
    
plt.figure(1,figsize=(12,9))  # 连续型变量相关图
corr = train3[number_feature+['SalePrice']].corr()
sns.heatmap(corr)

plt.figure(2,figsize=(12,9))  # 类别型变量相关图（离散型和伪数值型变量均已被概括为等级型变量）
corr = train3[category_E+['SalePrice']].corr('spearman')
sns.heatmap(corr)

plt.figure(3,figsize=(12,9)) # 连续型变量-类别型变量相关图
corr = pd.DataFrame(np.zeros([len(number_feature)+1, len(category_E)+1]), 
                    index=number_feature+['SalePrice'], columns=category_E+['SalePrice'])
for q1 in number_feature+['SalePrice']:
    for q2 in category_E+['SalePrice']:
        corr.loc[q1, q2] = train3[q1].corr(train3[q2], 'spearman')
sns.heatmap(corr)
```
- 通过上面两种图像，我们可以对转换过后的所有特征做一些筛选。

#### 3.把数值型特征转化为类别型特征

同上面相反，如果有些数值特征和我们的目标变量没有明显的线性关系，对于此类特征，我们应该将其转化为类别特征
```python
train[feature1] = train[feature1].astype(str)
train[feature2] = train[feature2].astype(str)
train[feature3] = train[feature3].astype(str)
train[feature4] = train[feature4].astype(str)
```
- 在所有的特征转化完成后，记得用 One - Hot 编码处理数据才能训练。



#### 4.删除离散样本点
我们可以通过散点图发现那些离散点，进而删除该样本点。
```python
#绘制散点图
import matplotlib.pyplot as plt
output,var,var1,var2 = 'SalePrice','GrLivArea','TotalBsmtSF','OverallQual'
fig,axis = plt.subplots(1,3,figsize=(16,5))
train.plot.scatter(x=var,y=output,ylim=(MIN_VALUE,MAX_VALUE),ax=axis[0])
train.plot.scatter(x=var1,y=output,ylim=(MIN_VALUE,MAX_VALUE),ax=axis[1])
train.plot.scatter(x=var2,y=output,ylim=(MIN_VALUE,MAX_VALUE),ax=axis[2])

#删除离散样点
train.drop(train[(train['Feature']>4000)&(train.SalePrice<300000)].index,inplace=True)
```
#### 5.删除无关特征

- 在我们的数据中，有可能会存在一些明显与目标变量无关的特征，对于这些特征我们应该删除，以免对我们之后的模型训练造成影响

#### 6.添加新的特征

- 在我们做项目的过程中还会发现，现有的特征数目太少，或者根据现有的特征训练出来的模型效果不理想。那么这个时候就需要我们结合实际项目，想办法构建出新的特征，进而提高我们模型的效果。

