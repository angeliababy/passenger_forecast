# -*- coding:utf-8 -*-
# 客流预测模型
# 实现resnet卷积网络
# 分(考虑和)不考虑外部条件两种情况

import tensorflow as tf

# 数据尺寸
size1 = 16
size2 = 8

# weight
def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

# bias
def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

# conv
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
#
# def dropout(x, keep):
#     return tf.nn.dropout(x, keep)

# 朴素残差单元模块
def residual(inputs, in_filter, out_filter, is_training):
    shortcut = inputs
    # 卷积层
    W1 = weightVariable([3, 3, in_filter, out_filter])
    b1 = biasVariable([out_filter])
    conv1 = conv2d(inputs, W1) + b1

    # BN层
    bn1 = tf.layers.batch_normalization(conv1, axis=3, training=is_training)
    # ReLu层
    relu1 = tf.nn.relu(bn1)

    # 卷积层
    W2 = weightVariable([3, 3, out_filter, out_filter])
    b2 = biasVariable([out_filter])
    conv2 = conv2d(relu1,W2) + b2

    # BN层
    bn2 = tf.layers.batch_normalization(conv2, axis=3, training=is_training)

    # 合并残差层
    # 当通道数有变化时
    if in_filter != out_filter:
        W_shortcut = weightVariable([1, 1, in_filter, out_filter])
        shortcut = tf.nn.conv2d(shortcut, W_shortcut, strides=[1, 1, 1, 1], padding='VALID')
    inputs = bn2 + shortcut
    # ReLu层
    inputs = tf.nn.relu(inputs)
    return inputs

# CNN模型结构
def cnnLayer(x, is_training):
    # 第一层卷积
    W1 = weightVariable([3,3,int(x.shape[3]),64]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = biasVariable([64])
    x = tf.nn.relu(conv2d(x, W1) + b1)
    # # 减少过拟合，随机让某些权重不更新
    # drop1 = dropout(pool1, keep_prob_5)

    # 4组朴素残差单元
    x = residual(x, 64, 64, is_training)
    x = residual(x, 64, 64, is_training)
    #
    x = residual(x, 64, 128, is_training)
    x = residual(x, 128, 128, is_training)
    #
    x = residual(x, 128, 256, is_training)
    x = residual(x, 256, 256, is_training)
    #
    x = residual(x, 256, 512, is_training)
    x = residual(x, 512, 512, is_training)

    # 卷积层
    W1 = weightVariable([3, 3, 512, 1])  # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = biasVariable([1])
    out = tf.nn.relu(conv2d(x, W1) + b1)
    out = tf.reshape(out, [-1, size1, size2])

    return out

# 融合层
def fusion(x1, x2, x3, is_training):
    # 三组件调用同一个卷积模型
    out1 = cnnLayer(x1, is_training)
    out2 = cnnLayer(x2, is_training)
    out3 = cnnLayer(x3, is_training)
    W1 = weightVariable([size1, size2])
    W2 = weightVariable([size1, size2])
    W3 = weightVariable([size1, size2])
    # 短期、周期、趋势融合
    out = tf.multiply(out1,W1) + tf.multiply(out2,W2) + tf.multiply(out3,W3)
    # tanh控制输出范围到(-1,1)
    out = tf.nn.tanh(out)
    return out