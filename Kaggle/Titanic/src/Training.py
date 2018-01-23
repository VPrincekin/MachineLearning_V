#!/usr/bin/python
#coding = utf-8
import pandas as pd
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

    # """KNN"""
    # print("===================KNN======================")
    # from sklearn.neighbors import KNeighborsClassifier
    # knn = KNeighborsClassifier(n_neighbors=10)
    # knn.fit(TrainData,TrainLabel)
    # acc_knn1 = round(knn.score(TrainData, TrainLabel) * 100, 2)
    # print(acc_knn1)
    # acc_knn2 = round(knn.score(TestData, TestLabel) * 100, 2)
    # print(acc_knn2)
    # Y_pred = knn.predict_proba(TestData)[:,1]
    # acc_knn3 = roc_auc_score(TestLabel,Y_pred)
    # print(acc_knn3)

    # """决策树"""
    # print("===================决策树======================")
    # from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
    #
    # dect = DecisionTreeClassifier()
    # dect.fit(TrainData,TrainLabel)
    # acc_dect1 = round(dect.score(TrainData, TrainLabel) * 100, 2)
    # print(acc_dect1)
    # acc_dect2 = round(dect.score(TestData, TestLabel) * 100, 2)
    # print(acc_dect2)
    #
    # """Logistic回归"""
    # print("====================Logistic回归==================")
    # from sklearn.linear_model import LogisticRegression
    # log = LogisticRegression()
    # log.fit(TrainData,TrainLabel)
    # acc_log1 = round(log.score(TrainData,TrainLabel)*100,2)
    # print(acc_log1)
    # acc_log2 = round(log.score(TestData,TestLabel)*100,2)
    # print(acc_log2)

    # """SVM"""
    # print("====================SVM===============================")
    # from sklearn.svm import SVC
    # svc = SVC(C=1.5,kernel='rbf')
    # svc.fit(TrainData,TrainLabel)
    # acc_svc1 = round(svc.score(TrainData,TrainLabel)*100,2)
    # print(acc_svc1)
    # acc_svc2 = round(svc.score(TestData,TestLabel)*100,2)
    # print(acc_svc2)

    # """Adaboost"""
    # print("====================Adaboost===============================")
    # from sklearn.ensemble import AdaBoostClassifier
    # ada = AdaBoostClassifier(n_estimators=100,learning_rate=0.5)
    # ada.fit(TrainData,TrainLabel)
    # acc_ada1 = round(ada.score(TrainData, TrainLabel) * 100, 2)
    # print(acc_ada1)
    # acc_ada2 = round(ada.score(TestData, TestLabel) * 100, 2)
    # print(acc_ada2)

    """XGBoost"""
    print("====================XGBoost===============================")
    from xgboost import XGBClassifier
    xgb = XGBClassifier(max_depth=5,learning_rate=0.4, gamma=0.03)
    # xgb.fit(TrainData, TrainLabel)
    # acc_xgb1 = round(xgb.score(TrainData, TrainLabel) * 100, 2)
    # print(acc_xgb1)
    # acc_xgb2 = round(xgb.score(TestData, TestLabel) * 100, 2)
    # print(acc_xgb2)
    xgb.fit(X_train,Y_train)
    print(round(xgb.score(X_train,Y_train),2))
    XGB_pred = xgb.predict(X_test)
    XGB_sub = pd.DataFrame({"PassengerId":test_df["PassengerId"],"Survived": XGB_pred})
    XGB_sub.to_csv('../data/XGB_sub.csv',index=False)

    """RandomForest"""
    print("====================RandomForest===============================")
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=15,max_depth=15)
    # rfc.fit(TrainData, TrainLabel)
    # acc_rfc1 = round(rfc.score(TrainData, TrainLabel) * 100, 2)
    # print(acc_rfc1)
    # acc_rfc2 = round(rfc.score(TestData, TestLabel) * 100, 2)
    # print(acc_rfc2)
    rfc.fit(X_train,Y_train)
    print(round(rfc.score(X_train, Y_train), 2))
    RFC_pred = rfc.predict(X_test)
    RFC_sub = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": RFC_pred})
    RFC_sub.to_csv('../data/RFC_sub.csv', index=False)

    """GBDT"""
    print("====================GBDT===============================")
    from sklearn.ensemble import GradientBoostingClassifier
    gbc = GradientBoostingClassifier(learning_rate=0.5,max_depth=5)
    # gbc.fit(TrainData, TrainLabel)
    # acc_gbc1 = round(gbc.score(TrainData, TrainLabel) * 100, 2)
    # print(acc_gbc1)
    # acc_gbc2 = round(gbc.score(TestData, TestLabel) * 100, 2)
    # print(acc_gbc2
    gbc.fit(X_train,Y_train)
    print(round(gbc.score(X_train, Y_train), 2))
    GBDT_pred = gbc.predict(X_test)
    GBDT_sub = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": GBDT_pred})
    GBDT_sub.to_csv('../data/GBDT_sub.csv', index=False)