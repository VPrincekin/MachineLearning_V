# !/usr/bin/env python
#coding = utf-8
from importModule import *
"""
方法参数说明
    trainData:  type: DataFrame 训练数据集
    testData:   type: DataFrame 测试数据集
    columns:    type: String    指定特征的列名
    columnsArr: type: List      指定多个特征的列名组成的集合
    Target:     type: List      目标值的列名
"""
class DataAna():
    def __init__(self,trainData,testData):
        self.trainDf = pd.DataFrame(trainData)
        self.testDf = pd.DataFrame(testData)
        self.data_clean = [self.trainDf,self.testDf]

    def dataInfo(self):
        logging.info('------训练数据信息')
        self.trainDf.info()
        self.trainDf.sample(10)

    def dataNull(self):
        logging.info('------查看训练/测试数据每列有多少个空字段')
        print('Train columns with null valules :\n', self.trainDf.isnull().sum())
        print('Test/Validation columns with null values:\n', self.testDf.isnull().sum())

    def dataDescribe(self):
        logging.info('------查看训练数据总体情况')
        self.trainDf.describe(include='all')

    def dataReplace(self,columns,method):
        for dataset in self.data_clean:
            if method == 'median':
                logging.info('------使用中位数替换')
                dataset[columns].fillna(dataset[columns].median(),inplace=True)
            elif method == 'mode':
                logging.info('------使用众数替换')
                dataset[columns].fillna(dataset[columns].mode()[0],inplace=True)
            elif method == 'mean':
                logging.info('------使用平均数替换')
                dataset[columns].fillna(dataset[columns].mean(),inplace=True)

    def dataDelete(self,cloumnsArr):
        logging.info('------删除指定列')
        self.trainDf.drop(cloumnsArr,axis=1,inplace=True)

    def dataTransform(self,columns):
        logging.info('------把一些非数值特征转化为数值特征')
        label = LabelEncoder()
        for dataset in self.data_clean:
            dataset[columns+'_Code'] = label.fit_transform(dataset[columns])

    def dataSplit(self,columnsArr,Target):
        logging.info('------拆分数据集')
        train_x_bin,test_x_bin,train_y_bin,test_y_bin = \
            model_selection.train_test_split(self.trainDf[columnsArr],self.trainDf[Target],random_state=0)
        return train_x_bin,test_x_bin,train_y_bin,test_y_bin

    def dataGroupAna(self,columnsArr,Target):
        logging.info('------指定分组分析数据')
        for x in columnsArr:
            if self.trainDf[x].dtype != 'float64':
              logging.info('{} Correlation by : {}'.format(Target,x))
              print(self.trainDf[x,Target].groupby(x,as_index=False).mean())
              print('-'*25)

        #print(pd.crosstab(data1['Title'], data1[Target[0]]))


