#!/usr/bin/python
#coding = utf-8
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
if __name__ == '__main__':
    #经过特征工程处理的数据集
    test_df = pd.read_csv('../data/test.csv')
    X_train = pd.read_csv('../data/X_train.csv')
    Y_train = pd.read_csv('../data/Y_train.csv',header=None)
    X_test = pd.read_csv('../data/X_test.csv')
    TrainData = X_train.loc[:700,:]
    TrainLabel = Y_train.loc[:700,:]
    TestData = X_train.loc[701:,:]
    TestLabel = Y_train.loc[701:,:]

    """交叉验证"""
    from sklearn import model_selection
    cvSplit = model_selection.KFold(10)


    # """随机森林"""
    # from  sklearn.ensemble import RandomForestClassifier
    # rf = RandomForestClassifier(n_estimators=150,min_samples_leaf=3,max_depth=6,oob_score=True)
    # #模型训练
    # cv_results = model_selection.cross_validate(rf,X_train,Y_train,cv=cvSplit)
    # print(cv_results["train_score"].mean(),cv_results["test_score"].mean())
    # #保存模型
    # from sklearn.externals import  joblib
    # joblib.dump(rf,"rf.m")
    # #恢复模型
    # rf_load = joblib.load("rf.m")
    # #预测结果
    # # rf_pre = rf.predict(X_test)
    # # rf_sub = pd.DataFrame({"PassengerId":test_df["PassengerId"],"Survived": rf_pre})
    # # rf_sub.to_csv()
    #
    #
    # """Logistic回归"""
    # from sklearn.linear_model import LogisticRegression
    # lr = LogisticRegression()
    # param = {"C":[0.1,0.5,0.8,1,10],"max_iter":[100,200,300]}
    # #自动选择最优参数
    # clf = model_selection.GridSearchCV(lr,param,scoring="roc_auc",cv=cvSplit)
    # clf.fit(X_train,Y_train)
    # #打印最佳得分和最佳参数
    # print(clf.best_score_,clf.best_params_)
    # #预测结果
    # # clf.predict(X_test)
    # # clf_sub = pd.DataFrame({"PassengerId":test_df["PassengerId"],"Survived": rf_pre})
    # # clf_sub.to_csv()

    # """模型融合-Voting"""
    # from sklearn.ensemble import VotingClassifier
    #
    # from sklearn.linear_model import LogisticRegression
    # lr = LogisticRegression(C=0.5,max_iter=100)
    #
    # import xgboost as xgb
    # xgb_model = xgb.XGBClassifier(max_depth=6,n_estimators=100)
    #
    # from sklearn.ensemble import RandomForestClassifier
    # rf = RandomForestClassifier(n_estimators=200,min_samples_leaf=2,max_depth=6,oob_score=True)
    #
    # from sklearn.ensemble import GradientBoostingClassifier
    # gbdt = GradientBoostingClassifier(learning_rate=0.1,min_samples_leaf=2,max_depth=6,n_estimators=100)
    #
    # vot = VotingClassifier(estimators=[('lr',lr),('rf',rf),('gbdt',gbdt),('xgb',xgb_model)],voting='hard')
    # vot.fit(X_train,Y_train)
    # vot.predict(X_test)
    # print(round(vot.score(X_train,Y_train)*100,2))
    #

    """模型融合-Stacking"""
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    clfs = [LogisticRegression(C=0.5, max_iter=100),
            xgb.XGBClassifier(max_depth=6, n_estimators=100, num_round=5),
            RandomForestClassifier(n_estimators=100, max_depth=6, oob_score=True),
            GradientBoostingClassifier(learning_rate=0.3, max_depth=6, n_estimators=100)]
    clf2 = LogisticRegression(C=0.5,max_iter=100)
    from mlxtend.classifier import StackingClassifier,StackingCVClassifier
    sclf = StackingClassifier(classifiers=clfs,meta_classifier=clf2)
    sclf.fit(X_train,Y_train)
    print(sclf.score(X_train,Y_train))
    sclf2 = StackingCVClassifier(classifiers=clfs,meta_classifier=clf2,cv=3)
    x = np.array(X_train)
    y = np.array(Y_train).flatten()
    sclf2.fit(x,y)
    print(sclf2.score(x,y))

