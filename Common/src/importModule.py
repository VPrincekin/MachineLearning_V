# !/use/bin/env python
# coding = utf-8

import warnings
warnings.filterwarnings('ignore')

"""基础模块"""
import sys
import pandas as pd
import numpy as np
import random
import time

"""公共算法库"""
from sklearn import svm,tree,linear_model,neighbors,naive_bayes,ensemble,discriminant_analysis,gaussian_process
from xgboost import XGBClassifier

"""辅助模块"""
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

"""可视化模块"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns


"""配置可视化设置"""
# %matplotlib inline 在jupyter notebook上配置
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
