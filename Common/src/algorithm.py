#!/usr/bin/env python
#coding=utf-8
from importModule import *
"""
方法参数说明
    trainDF：    type：DataFrame  训练数据集
    featureArr： type：List       训练集所有特征的列名组成的集合
    Target：     type：List       目标值的列名      
    alg:         type: MLA        指定的算法
"""
class MLA():
    def __init__(self):
        logging.info('------MLA初始化')
        self.MLA = [
        # 集成方法
        ensemble.AdaBoostClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier(),
        ensemble.ExtraTreesClassifier(),

        # 高斯过程
        gaussian_process.GaussianProcessClassifier(),

        # 线性模型
        linear_model.LogisticRegressionCV(),
        linear_model.PassiveAggressiveClassifier(),
        linear_model.RidgeClassifierCV(),
        linear_model.SGDClassifier(),
        linear_model.Perceptron(),

        # 贝叶斯
        naive_bayes.GaussianNB(),
        naive_bayes.BernoulliNB(),

        # 近邻算法
        neighbors.KNeighborsClassifier(),

        # SVM
        svm.SVC(probability=True),
        svm.LinearSVC(),
        svm.NuSVC(),

        # Trees
        tree.DecisionTreeClassifier(),
        tree.ExtraTreeClassifier(),

        # 判别分析
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),

        # XGBoost
        XGBClassifier()
            ]
        #使用拆分器交叉验证数据集，这是model_selection.train_test_split()的替代方案
        self.cvSplit = model_selection.ShuffleSplit(n_splits=10,test_size=.3,train_size=.6,random_state=0)
        self.MLA_columns = ['MLA_Name','MLA_Parameters','MLA Train Accuracy Mean','MLA Test Accuracy Mean','MLA Test Accuracy 3*STD','MLA Time']
        self.MLA_compare = pd.DataFrame(columns=self.MLA_columns)

    def predict(self,trainDF,featureArr,Target):
        logging.info('---查看各个模型的效果')
        MLA_predict = trainDF[Target]
        row_index = 0
        for alg in self.MLA:
            #设置名字和参数
            MLA_name = alg.__class__.__name__
            self.MLA_compare.loc[row_index,'MLA_Name'] = MLA_name
            self.MLA_compare.loc[row_index,'MLA_Parameters'] = str(alg.get_params())
            #交叉验证得分模型

            cv_results = model_selection.cross_validate(alg,trainDF[featureArr],trainDF[Target],cv=self.cvSplit)
            self.MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
            self.MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
            self.MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std() * 3
            self.MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

            #保存预测值
            alg.fit(trainDF[featureArr],trainDF[Target])
            MLA_predict[MLA_name] = alg.predict(trainDF[featureArr])
            row_index +=1

        self.MLA_compare.sort_values(by=['MLA Test Accuracy Mean'],ascending=False,inplace=True)
        print(self.MLA_compare)
        #通过图表展示效果
        sns.barplot(x='MLA Test Accuracy Mean',y='MLA_Name',data=self.MLA_compare,color='m')
        plt.xlabel('Accuracy Score(%)')
        plt.ylabel('Algorithm')
        plt.title('Machine Learning Algorithm Accuracy Score \n')
        plt.show()
        return MLA_predict

    def tuneFeature(self,trainDF,featureArr,Target,alg):
        logging.info('------自动选择特征')
        alg_rfe = feature_selection.RFECV(alg,step=1,scoring='accuracy',cv=self.cvSplit)
        alg_rfe.fit(trainDF[featureArr],trainDF[Target])

        X_rfe = trainDF[trainDF[featureArr].columns.values(alg_rfe.get_support())]
        rfe_results = model_selection.cross_validate(alg,trainDF[featureArr],trainDF[Target],cv=self.cvSplit)
        logging.info('------最好的特征以及该特征下模型效果')
        print('AFTER DT RFE Training Shape New: ', trainDF[X_rfe].shape)
        print('AFTER DT RFE Training Columns New: ', X_rfe)
        print("AFTER DT RFE Training w/bin score mean: {:.2f}".format(rfe_results['train_score'].mean() * 100))
        print("AFTER DT RFE Test w/bin score mean: {:.2f}".format(rfe_results['test_score'].mean() * 100))
        print("AFTER DT RFE Test w/bin score 3*std: +/- {:.2f}".format(rfe_results['test_score'].std() * 100 * 3))

    def tuneModel(self,trainDF,featureArr,Target,alg,paramdict):
        logging.info('------自动调节参数')
        tune_model = model_selection.GridSearchCV(alg,param_grid=paramdict,scoring='roc_auc',cv=self.cvSplit)
        tune_model.fit(trainDF[featureArr],trainDF[Target])

        logging.info('------最好好的模型参数以及该参数下模型效果')
        tune_results = model_selection.cross_validate(alg, trainDF[featureArr], trainDF[Target], cv=self.cvSplit)
        print('AFTER DT Parameters: ', tune_model.best_params_)
        print("AFTER DT Training w/bin set score mean: {:.2f}".format(tune_results['train_score'].mean() * 100))
        print("AFTER DT Test w/bin set score mean: {:.2f}".format(tune_results['test_score'].mean() * 100))
        print("AFTER DT Test w/bin set score min: {:.2f}".format(tune_results['test_score'].min() * 100))

    def thermogramModel(self,trainDF,featureArr,Target):
        logging.info('------热力图，观察不同模型之间的相关性')
        MLA_predict = self.predict(trainDF,featureArr,Target)
        import graphics
        Ga = graphics.GraphAna()
        Ga.correlation_heatmap(MLA_predict)



