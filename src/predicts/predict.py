# coding=utf-8
# 客流预测模型第三步：预测部分
# 单步预测，多步预测等真实数据继续处理

import tensorflow as tf
import pandas as pd
import numpy as np
import cv2

# 调用其他模块
import sys
sys.path.append('../data_pre')
import data
sys.path.append('../model_defin')
import model_resnet_no
sys.path.append('../train_models')
import external_model_no
# 数据尺寸
size1 = 16
size2 = 8

# 预测部分
def evaluate(merged1, merged2, merged3):
    x1 = tf.placeholder(tf.float32, [None, size1, size2, 3])
    x2 = tf.placeholder(tf.float32, [None, size1, size2, 3])
    x3 = tf.placeholder(tf.float32, [None, size1, size2, 3])
    is_training = tf.placeholder(tf.bool, name='is_training')
    y = model_resnet_no.fusion(x1, x2, x3, is_training)  # 预测结果

    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(noexternal_model.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

            # 单步预测
            predict_y = sess.run(y, feed_dict={x1:merged1, x2:merged2, x3:merged3, is_training:False})
            return predict_y
        else:
            print('No checkpoint file found')
            return

def main(argv=None):
    imgs = data.Preprocess()[8]
    # # 取最后一组进行预测
    # i = 1350
    # image1 = [imgs[i - 3], imgs[i - 2], imgs[i - 1]]
    # image2 = [imgs[i - 72], imgs[i - 48], imgs[i - 24]]
    # image3 = [imgs[i - 484], imgs[i - 336], imgs[i - 168]]
    # # 融合成多通道
    # merged1 = cv2.merge([image1[0], image1[1], image1[2]])
    # merged1 = merged1.reshape(1, size, size, 3)
    # merged2 = cv2.merge([image2[0], image2[1], image2[2]])
    # merged2 = merged2.reshape(1, size, size, 3)
    # merged3 = cv2.merge([image3[0], image3[1], image3[2]])
    # merged3 = merged3.reshape(1, size, size, 3)
    # # 单步预测未来一个时刻的数据
    # predict_y = evaluate(merged1, merged2, merged3)
    # for predict in predict_y:
    #     print((412640*predict).astype(int))

    test_x1, test_x2, test_x3 = [], [], []
    for i in range(1350, 1353):
    # 取短期、周期、趋势三组件数据
        image1 = [imgs[i - 3], imgs[i - 2], imgs[i - 1]]
        image2 = [imgs[i - 72], imgs[i - 48], imgs[i - 24]]
        image3 = [imgs[i - 484], imgs[i - 336], imgs[i - 168]]
        # 融合成多通道
        merged1 = cv2.merge([image1[0], image1[1], image1[2]])
        merged2 = cv2.merge([image2[0], image2[1], image2[2]])
        merged3 = cv2.merge([image3[0], image3[1], image3[2]])
        test_x1.append(merged1)
        test_x2.append(merged2)
        test_x3.append(merged3)
    predict_y = evaluate(test_x1, test_x2, test_x3)
    # print(predict_y)
    print((412640*predict_y))

if __name__ == '__main__':
    tf.app.run()