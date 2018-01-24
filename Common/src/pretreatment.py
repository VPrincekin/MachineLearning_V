# !/usr/bin/env python
#coding = utf-8
from importModule import *
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

class Data():
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

    def dataReplace(self,cloumns,method):
        for dataset in self.data_clean:
            if method == 'median':
                logging.info('------使用中位数替换')
                dataset[cloumns].fillna(dataset[cloumns].median(),inplace=True)
            elif method == 'mode':
                logging.info('------使用众数替换')
                dataset[cloumns].fillna(dataset[cloumns].mode()[0],inplace=True)
            elif method == 'mean':
                logging.info('------使用平均数替换')
                dataset[cloumns].fillna(dataset[cloumns].mean(),inplace=True)

    def dataDelete(self,cloumnsArr):
        logging.info('------删除指定列')
        self.trainDf.drop(cloumnsArr,axis=1,inplace=True)

    def dataTransform(self,cloumns):
        logging.info('------把一些非数值特征转化为数值特征')
        label = LabelEncoder()
        for dataset in self.data_clean:
            dataset[cloumns+'_Code'] = label.fit_transform(dataset[cloumns])

    def dataSplit(self,cloumnsArr,Target):
        logging.info('------拆分数据集')
        train_x_bin,test_x_bin,train_y_bin,test_y_bin = \
            model_selection.train_test_split(self.trainDf[cloumnsArr],self.trainDf[Target],random_state=0)
        return train_x_bin,test_x_bin,train_y_bin,test_y_bin

    def dataGroupAna(self,cloumnsArr,Target):
        logging.info('------指定分组分析数据')
        for x in cloumnsArr:
            if self.trainDf[x].dtype != 'float64':
              logging.info('{} Correlation by : {}'.format(Target,x))
              print(self.trainDf[x,Target].groupby(x,as_index=False).mean())
              print('-'*25)
        #print(pd.crosstab(data1['Title'], data1[Target[0]]))


