# -*- coding:utf-8 -*-
# 客流预测模型第二步：模型训练及保存模型
# 卷积模型训练部分(含外部条件)

import tensorflow as tf
import numpy as np
import cv2
import os
import argparse

# 导入数据
import sys
sys.path.append('../data_pre')
import data
sys.path.append('../model_defin')
import model_resnet

#训练轮数
TRAINING_STEPS = 100
MODEL_SAVE_PATH = '../../models/'
MODEL_NAME = "external_model"

# 数据尺寸
size = 3

# 模型训练
def train(train_x1, train_x2, train_x3, test_x1, test_x2, test_x3, train_y, test_y, batch_size, num_batch, train_ex, test_ex):
    # 定义网络输入
    ex = tf.placeholder(tf.float32, [None, train_ex.shape[1]])
    x1 = tf.placeholder(tf.float32, [None, size, size, train_x1.shape[3]])
    x2 = tf.placeholder(tf.float32, [None, size, size, train_x2.shape[3]])
    x3 = tf.placeholder(tf.float32, [None, size, size, train_x3.shape[3]])
    y_ = tf.placeholder(tf.float32, [None, size, size])

    # 是否训练
    is_training = tf.placeholder(tf.bool, name='is_training')
    # keep_prob_5 = tf.placeholder(tf.float32)
    # keep_prob_75 = tf.placeholder(tf.float32)

    # 模型输出
    y = model_resnet.exnetwork(ex, x1, x2, x3, is_training)

    # 采用平均误差平方和作为损失函数
    loss = tf.reduce_mean(tf.square(y_ - y))  # 定义平均误差平方作为损失函数
    tf.add_to_collection('losses', loss)
    loss = tf.add_n(tf.get_collection('losses'))

    # 测试集平方和误差
    loss_mse = tf.reduce_mean(tf.square(50*y_ - 50*y))

    # Adam优化器
    train_step = tf.train.AdamOptimizer(0.0002).minimize(loss)

    # 将loss保存以供tensorboard使用
    tf.summary.scalar('loss', loss)
    merged_summary_op = tf.summary.merge_all()

    # 存储事件文件
    summary_writer = tf.summary.FileWriter('../../tmp', graph=tf.get_default_graph())

    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)  # 检查点
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)  # 加载已存在的模型，从断点开始训练
        else:
            tf.global_variables_initializer().run()  # 初始化所有变量
        # 迭代的训练网络
        for n in range(TRAINING_STEPS):
            for i in range(num_batch):
                # 训练部分短期、周期、趋势批次数据及输出
                batch_ex = train_ex[i * batch_size :(i + 1) * batch_size]
                batch_x1 = train_x1[i * batch_size :(i + 1) * batch_size]
                batch_x2 = train_x2[i * batch_size: (i + 1) * batch_size]
                batch_x3 = train_x3[i * batch_size: (i + 1) * batch_size]
                batch_y = train_y[i*batch_size : (i+1)*batch_size]
                # 开始训练数据，同时训练三个变量，返回三个数据
                _, loss_value, summary = sess.run([train_step, loss, merged_summary_op],
                                           feed_dict={ex:batch_ex, x1:batch_x1, x2:batch_x2, x3:batch_x3, y_:batch_y, is_training:True})
                summary_writer.add_summary(summary, n * num_batch + i)
                # 打印损失
                # print(n*num_batch+i, loss_value)
                if (n*num_batch+i) % 10 == 0:  # 每隔10轮打印训练效果
                    print("经过 %d 轮迭代训练, 训练集上的损失函数值为 %g." % (n*num_batch+i, loss_value))
                    test_mse = sess.run(loss_mse, feed_dict={ex:test_ex, x1:test_x1, x2:test_x2, x3:test_x3, y_:test_y, is_training:False})
                    print("经过 %d 轮迭代训练,测试集误差平方为 %g " % (n*num_batch+i, test_mse))
                    if test_mse < 20:  # 将模型进行保存
                        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=(n*num_batch+i))

def main(argv=None):
    # 内部条件
    train_x1, train_x2, train_x3, test_x1, test_x2, test_x3, train_y, test_y = data.Preprocess()[:8]

    # 外部条件
    train_ex, test_ex = data.external()

    print('train size:%s, test size:%s' % (len(train_x1), len(test_x1)))
    # 图片块，每次取32张图片
    batch_size = 32
    num_batch = len(train_x1) // batch_size
    # 训练网络
    train(train_x1, train_x2, train_x3, test_x1, test_x2, test_x3, train_y, test_y, batch_size, num_batch, train_ex, test_ex)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    tf.app.run()


