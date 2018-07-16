# public_praise_prediction_yunyi
2018云移杯- 景区口碑评价分值预测   7/1186 “云南我来了”队伍

比赛地址：http://www.datafountain.cn/?u=7609847&&#/competitions/283/intro

对用户给出的评语，预测其相应的评分，评分范围0~5，实际上是一个情感预测的问题。

### 简介

分别使用lightgbm, ridge, mlp三种模型预测口碑分数，三种模型大约都在0.54左右，最后做加权平均。各模型尽可能强而不同。

- 数据预处理

  首先做中文分词，分别尝试了SnowNLP，jieba，各种分词工具效果相差不大，因此三种模型故意使用不同的分词工具。做word embedding时，有word2vec(doc2vec)和TF-IDF，经过尝试，在不同的模型上不同的embedding有较大的差异，比如mlp我们使用的是TF-IDF，而lightgbm我们使用的word2vec，个人感觉是不是word2vec本身就是神经网络捕获的特征，导致再用mlp做反而效果提升不明显。word2vec有更丰富的语义，但是在这种情感分类问题上，表现得并不非常突出。而且发现去除停用词有时会使得效果变差，有可能比如“！”这种也实际表达了一种强烈的情感，是一个强特。

- 模型

  将上面的embedding结果传到各个模型中，原先做5类分类，效果较差。后来改成预测0~5之间的数值，在这里用的一个小trick就是大于4.7，全部作为评价为5，会有小小的提升。模型就是不断尝试和调差的过程，包括使用GridSearch找最优参数。用的mlp是一个比较浅的全连接神经网络，192、64、64、1，情感预测的神经网络不需要特别复杂特别深。

### 文件结构

- lgb_dc.py: lightgbm 分值预测模型的实现
- mlp.py: mlp 分值预测模型的实现
- ridge.py: ridge (linear model)分值预测模型的实现

### 联系我

[cnmengnan@gmail.com](mailto:cnmengnan@gmail.com)

blog: [WinterColor blog](http://www.cnblogs.com/mengnan/)