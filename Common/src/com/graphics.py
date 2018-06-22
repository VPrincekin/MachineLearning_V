# !/usr/bin/env python
# coding =utf-8
from importModule import *

"""
 方法参数说明
     trainData： type：DataFrame  训练数据集      
     columns:    type：String     指定特征的列名      
     columnsArr: type: List       指定多个特征的列名组成的集合
     Target:     type: String     目标值的列名
 """
class GraphAna():

    def __init__(self,trainData):
        self.trainDF = pd.DataFrame(trainData)

    def histogram(self,columns,Target):
        logging.info('------根据指定特征绘制直方图')
        plt.figure(figsize=[16, 12])
        plt.subplot(111)
        plt.hist(x=[self.trainDF[self.trainDF[Target]==1][columns],self.trainDF[self.trainDF[Target]==0][columns]],stacked=True,color=['g','r'],label=['1','0'])
        plt.title('{} Histogram by {}'.format(columns,Target))
        plt.xlabel(columns)
        plt.ylabel(Target)
        plt.legend()
        plt.show()

    def barPlot(self,columns,Target):
        logging.info('------多变量特征柱状图')
        plt.figure(figsize=[16, 12])
        sns.barplot(x=columns,y=Target,data=self.trainDF)
        plt.show()


    def doubleFeature(self,columnsArr,Target):
        logging.info('------指定两个特征之间相关性分析')
        fig, axis = plt.subplots(1, 1, figsize=(14, 12))
        x = columnsArr[0]
        hue = columnsArr[1]
        sns.barplot(x=x,y=Target,hue=hue,data=self.trainDF)
        axis.set_title('{} VS {} {} Comparison'.format(x,hue,Target))
        plt.show()

    def distributed(self,columns,Target):
        plt.figure(figsize=[16, 12])
        logging.info('------查看目标值在指定特征下的分布情况')
        a = sns.FacetGrid(self.trainDF,hue=Target,aspect=4)
        a.map(sns.kdeplot,columns,shade = True)
        a.set(xlim = (0, self.trainDF[columns].max()))
        a.add_legend()
        plt.show()


    def allFeature(self,Target):
        plt.figure(figsize=[14,12])
        logging.info('------在整个数据集上分析特征之间的相关性')
        pp = sns.pairplot(self.trainDF,hue=Target,size=1.5)
        pp.set(xticklabels=[])
        plt.show()


    def thermogram(self):
        logging.info('------特征相关性热力图')
        self.correlation_heatmap(self.trainDF)

    def correlation_heatmap(self,df):
        _, ax = plt.subplots(figsize=(14, 12))
        colormap = sns.diverging_palette(220, 10, as_cmap=True)

        _ = sns.heatmap(
            df.corr(),
            cmap=colormap,
            square=True,
            cbar_kws={'shrink': .9},
            ax=ax,
            annot=True,
            linewidths=0.1, vmax=1.0, linecolor='white',
            annot_kws={'fontsize': 12}
        )
        plt.title('Pearson Correlation of Features', y=1.05, size=15)
        plt.show()
