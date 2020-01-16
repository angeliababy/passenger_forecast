# -*- coding: utf-8 -*-
# 客流预测第一步
# 读取数据，数据预处理部分,将数据处理成图像多通道的形式

import numpy as np
import cv2
import math
import random
import pandas as pd
from sklearn.model_selection import train_test_split
size = 10

# 将短期、周期、趋势数据单独提出(此时没用到)
def separate(x):
    x1, x2, x3 = [], [], []
    for i in range(len(x)):
        x1.append(x[i][0])
        x2.append(x[i][1])
        x3.append(x[i][2])
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    return x1, x2, x3

# 数据预处理
def Preprocess():
    # 读取数据文件
    f = open("../../data/area.csv", "r")
    # 临时存储某时间的人数
    person_num = []
    # 存储各时间的人数尺寸(n,3,3)
    imgs = []

    i, l = 0, 0
    for line in f:
        l += 1
        if l == 1:
            continue
        i += 1
        line = line.strip().split(',')
        # 将人数转化为小于1的数，后面求实际人数需转化过来
        number = (float(line[2])-0) / (3073-0) * 2 - 1
        person_num.append(number)
        # 每次读16个数据
        if i % (128) == 0:
            # 转化成一维数组
            person_num = np.array(person_num)
            # 改变形状，类似图像形式
            person_num = person_num.reshape(16, 8)
            imgs.append(person_num)
            i = 0
            person_num = []

    # # 训练数据（输入三种类型的数据，并各自转化为多通道形式）
    # train_x1, train_x2, train_x3, train_y = [], [], [], []
    # for i in range(484, 1500):
    # # 取短期、周期、趋势三组件数据，各不同长度序列
    #     image1 = [imgs[i - 3], imgs[i - 2], imgs[i - 1]]
    #     image2 = [imgs[i - 72], imgs[i - 48], imgs[i - 24]]
    #     image3 = [imgs[i - 484], imgs[i - 336], imgs[i - 168]]
    #     # 融合成多通道
    #     merged1 = cv2.merge([image1[0], image1[1], image1[2]])
    #     merged2 = cv2.merge([image2[0], image2[1], image2[2]])
    #     merged3 = cv2.merge([image3[0], image3[1], image3[2]])
    #     train_x1.append(merged1)
    #     train_x2.append(merged2)
    #     train_x3.append(merged3)
    #     lab = imgs[i]
    #     train_y.append(lab)  # 最终输出
    #
    # # 测试数据（输入三种类型的数据，并各自转化为多通道形式）
    # test_x1, test_x2,test_x3, test_y = [], [], [], []
    # for i in range(1500, 1800):
    #     # 取短期、周期、趋势三组件数据
    #     image1 = [imgs[i - 3], imgs[i - 2], imgs[i - 1]]
    #     image2 = [imgs[i - 72], imgs[i - 48], imgs[i - 24]]
    #     image3 = [imgs[i - 484], imgs[i - 336], imgs[i - 168]]
    #     # 融合成多通道
    #     merged1 = cv2.merge([image1[0], image1[1], image1[2]])
    #     merged2 = cv2.merge([image2[0], image2[1], image2[2]])
    #     merged3 = cv2.merge([image3[0], image3[1], image3[2]])
    #     test_x1.append(merged1)
    #     test_x2.append(merged2)
    #     test_x3.append(merged3)
    #     lab = imgs[i]
    #     test_y.append(lab)  # 最终输出
    #
    # train_x1 = np.array(train_x1)
    # train_x2 = np.array(train_x2)
    # train_x3 = np.array(train_x3)
    # test_x1 = np.array(test_x1)
    # test_x2 = np.array(test_x2)
    # test_x3 = np.array(test_x3)
    # train_y = np.array(train_y)
    # test_y = np.array(test_y)

    images,labs = [],[]
    # 下面是内部数据随机生成训练、测试集的过程，但是如果用到外部数据必须要一一对应随机生成，相对麻烦，故此时没用
    for i in range(484, 1350):
        # 取短期、周期、趋势三组件数据
        image1 = [imgs[i - 3], imgs[i - 2], imgs[i - 1]]
        image2 = [imgs[i - 72], imgs[i - 48], imgs[i - 24]]
        image3 = [imgs[i - 484], imgs[i - 336], imgs[i - 168]]
        # 融合成多通道
        merged1 = cv2.merge([image1[0], image1[1], image1[2]])
        merged2 = cv2.merge([image2[0], image2[1], image2[2]])
        merged3 = cv2.merge([image3[0], image3[1], image3[2]])
        # 组合list
        image = [merged1, merged2, merged3]
        images.append(image)  # 短期、周期、趋势数据在一起
        lab = imgs[i]
        labs.append(lab)  # 最终输出
    labs = np.array(labs)

    # 随机生成训练、测试集
    train_x, test_x, train_y, test_y = train_test_split(images, labs, test_size=0.1,
                                                        random_state=random.randint(0, 100))

    # 分离训练集中的短期、周期、趋势数据
    train_x1, train_x2, train_x3 = separate(train_x)
    # 分离测试集中的短期、周期、趋势数据
    test_x1, test_x2, test_x3 = separate(test_x)

    return train_x1, train_x2, train_x3, test_x1, test_x2, test_x3, train_y, test_y, imgs

# # 外部条件数据
# def external():
#     from sklearn.utils import shuffle
#     # 读取数据
#     datas = pd.read_csv("../../data/external.csv", "r")
#     # 打乱数据（此时因为是虚构数据）
#     datas = shuffle(datas)
#     X = datas[["week", "weather", "tem"]]
#     X['week'] = X['week'].astype(str)
#     X['weather'] = X['weather'].astype(str)
#
#     # 最大最小值归一化
#     X['tem'] = X['tem'].transform(
#         lambda x: (x - x.min()) / x.max())
#
#     # 数据特征转换：将文本特征转换为数值特征
#     from sklearn.feature_extraction import DictVectorizer
#     vec = DictVectorizer(sparse=False)
#     X = vec.fit_transform(X.to_dict(orient='record'))
#
#     # 训练、测试数据，与内部数据取的要对应
#     train_ex = X[:900]
#     test_ex = X[900:]
#
#     # # 数据标准化
#     # from sklearn.preprocessing import StandardScaler
#     # ss_X = StandardScaler()
#     # train_ex = ss_X.fit_transform(train_ex)
#     # test_ex = ss_X.transform(test_ex)
#     return train_ex, test_ex