# 利用深度时空残差网络预测城市范围的交通流量
本项目完整源码地址：[https://github.com/angeliababy/passenger_forecast](https://github.com/angeliababy/passenger_forecast)

项目博客地址: [https://blog.csdn.net/qq_29153321/article/details/104005743](https://blog.csdn.net/qq_29153321/article/details/104005743)


## 原理部分
参考论文《Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction》或
《利用深度时空残差网络预测城市范围的人流量》

下载地址：[https://download.csdn.net/download/qq_29153321/10998326](https://download.csdn.net/download/qq_29153321/10998326)

1.解决问题：城市人流量预测

2.应用方法：ST-ResNet（卷积残差网络）

 ![在这里插入图片描述](https://img-blog.csdnimg.cn/2020011517281969.png)
 
应用方法优势：建模近处的和远处的两个区域之间的空间依赖性，同时也保证了预测的精度不受神经网络的深度结构影响。

3.人流量：将城市分为M*N网格形式，计算进入流和外出流
    时间依赖性：短期、周期、趋势
    外部因素影响：天气、工作日情况等
    
4.下图展示了 ST-ResNet 的框架，包括四个主要的模块：分别建模邻近性、周期性、趋势性和外部影响因子（暂不考虑）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200115172916705.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MTUzMzIx,size_16,color_FFFFFF,t_70) 

框架解析：暂时不考虑图中外部因素影响融合部分。比如以短期序列为例，令最近的一个分段为![\[Xt-lc,Xt-(lc-1),…,Xt-1\]](https://img-blog.csdnimg.cn/2020011615213077.png)
,称为邻近性依赖矩阵，将它们与时间轴个数进行拼接成张量 
![Xc(0) ∈R2lc×I×J](https://img-blog.csdnimg.cn/20200116152149116.png)
，周期、趋势序列中Conv1-Conv2过程类似输出
![〖X_p〗^((L+2))、〖X_q〗^((L+2))](https://img-blog.csdnimg.cn/20200116152201773.png)
.其中卷积采用边界填充0尺寸不变卷积形式.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200116151951641.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MTUzMzIx,size_16,color_FFFFFF,t_70)

5.具体操作如下：
数据准备：将城市划分为M*N网格，计算每个网格上基于时间依赖性的进入流和外出流。
数据输入：以短期预测为例，尺寸大小（M，N，2lc），前两维表示网格尺寸，后一维表示各时间节点的进入流、外出流。其中卷积操作和残差单元示例如下，参数数目L可变,此时为4，Adam自优化学习率，batch_size=32, p和 q分别设为 1 天和一周。对于 3 个序列，设为：lc∈{3,4,5}，lp ∈{1,2,3,4}，lq∈{1,2,3,4}。90%数据训练，10%数据测试，迭代10100次。 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200116152336287.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MTUzMzIx,size_16,color_FFFFFF,t_70)

6.流程图（输入数据会做tanh归一化预处理）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200116152524649.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MTUzMzIx,size_16,color_FFFFFF,t_70)

## 实践部分：resnet卷积网络解决客流预测问题

首先将城市的客流数据依据区域转化为图像像素的形式，然后利用resnet卷积网络对具有时间序列性质的客流数据（转化为图像通道）进行未来时刻的区域客流预测。

### 一、各目录情况如下：
#### datas
    目录下存储内部条件序列数据（data.csv）和外部条件序列数据(external.csv)，其他为客流数据接入用的虚构数据

#### src
    1. date_pre
    目录下data.py为客流预测的预处理过程（模型第一步）
    keliu.py为客流数据的接入过程（最先步骤）
    2. model_defin
    目录下定义resnet卷积网络结构定义（分为考虑外部条件external_model.py和不考虑外部条件external_model_no.py两种情况）
    3. train_models
    目录下实现客流预测的训练过程（模型第二步）（有两种情况：分为考虑外部条件和不考虑外部条件的网络模型训练）
    4. predicts
    目录下为客流预测的预测过程（模型第三步），此时为单步预测，之后有真实数据时需加入多步预测计算

#### models
    目录下会存储网络生成的模型文件
    
### 二、运行（以不考虑外部条件为例）
    1. 第一步，模型数据预处理，运行src/data_pre/data.py
    2. 第二步，模型训练操作，运行src/train_models/external_model_no.py
    3. 第三步，模型预测（单步预测），运行src/predicts/predict.py

最后需要注意的是：接入真实数据后需测试考虑外部条件和不考虑外部条件的情况，观察是否考虑外部条件的结果更好


