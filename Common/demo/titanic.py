#!/usr/bin/env python
#conding = utf-8

from com.importModule import *

data_raw = pd.read_csv('../data/train.csv')
data_val = pd.read_csv('../data/test.csv')
data_cleaner = [data_raw,data_val]
pt = pretreatment.DataAna(data_raw, data_val)
# pt.dataInfo() #训练数据信息
# pt.dataDescribe() #查看训练数据总体情况
# pt.dataNull()
pt.dataReplace('Age','median')
pt.dataReplace('Embarked','mode')
pt.dataReplace('Fare','mean')
pt.dataDelete(['PassengerId','Cabin','Ticket']) #删除特征
for dataset in pt.data_clean:
    #将SliSp和Parch两个特征，合并成一个特征：FamilySize
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    #根据FamilySize的大小，创建一个新的特征：IsAlone
    dataset['IsAlone'] = 1
    dataset['IsAlone'].loc[dataset['FamilySize']>1] = 0
    #切分人名，创建特征：Title
    dataset['Title'] = dataset['Name'].str.split(',',expand = True)[1].str.split('.',expand = True)[0]
    #基于分位数的离散化函数。将变量分为基于等级或基于样本分位数的相等大小的桶。
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

Rare_name = pt.trainDf['Title'].value_counts()<10
pt.trainDf['Title'] = pt.trainDf['Title'].apply(lambda x:'Rare' if Rare_name.loc[x] == True else x)

pt.dataTransform('Sex')
pt.dataTransform('Embarked')
pt.dataTransform('Title')
pt.dataTransform('AgeBin')
pt.dataTransform('FareBin')

columnsArr = ['Sex_Code','Pclass','Title_Code','AgeBin_Code','Embarked_Code','FareBin_Code','FamilySize','IsAlone']
Target = ['Survived']

pt.dataNull()
# pt.dataGroupAna(['Pclass'],'Survived')

Gra = graphics.GraphAna(pt.trainDf)
# Gra.thermogram()
# Gra.histogram('Fare','Survived')
# Gra.barPlot('Sex','Survived')
# Gra.doubleFeature(['Sex','Pclass'],'Survived')
# Gra.distributed('Age','Survived')
# Gra.allFeature('Survived')

from com import algorithm, graphics, pretreatment, stackingModel

# MLA = algorithm.MLA()
# MLA.predict(pt.trainDf,columnsArr,Target)
# MLA.thermogramModel(pt.trainDf,columnsArr,Target)


clf2 = svm.SVC(probability=True)
clf3 = ensemble.AdaBoostClassifier()
clf4 = ensemble.GradientBoostingClassifier()
clf5 = tree.DecisionTreeClassifier()
clfArr = [clf2,clf3,clf4,clf5]

stack = stackingModel.StackModel(clfArr)
pt.testDf['Survived'] = stack.stack(pt.trainDf[columnsArr],pt.trainDf[Target],pt.testDf[columnsArr])
submit = pt.testDf[['PassengerId','Survived']]
submit.to_csv('../data/submit.csv',index=False)
