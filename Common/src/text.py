#coding = utf-8
from importModule import *
from sklearn.model_selection import StratifiedKFold,KFold

X=np.array([
    [1,2,3,4],
    [11,12,13,14],
    [21,22,23,24],
    [31,32,33,34],
    [41,42,43,44],
    [51,52,53,54],
    [61,62,63,64],
    [71,72,73,74]
])

y=np.array([1,1,0,0,1,1,0,0])
#n_folds这个参数没有，引入的包不同，
floder = KFold(n_splits=5,random_state=0,shuffle=False)
sfolder = StratifiedKFold(n_splits=3,random_state=0,shuffle=False)
# StratifiedKFold 分层采样交叉切分，确保训练集，测试集中各类别样本的比例与原始数据集中相同。

for train, test in sfolder.split(X,y):
    print("SKF")
    print('Train: %s | test: %s' % (train, test))


for train, test in floder.split(X,y):
    print("KF")
    print('Train: %s | test: %s' % (train, test))
