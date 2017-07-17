#coding=utf-8

import tensorflow as tf
import time

"""
自动下载并导入MNIST数据集  每张图28*28 
特征表示在导入时已经处理好
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 标签采用one_hot表示

x = tf.placeholder(tf.float32, [None, 784])   # 把图展开成一维向量
y = tf.placeholder(tf.float32, [None, 10])

"""
保证输入输出是同一个大小：
    卷积的步长（stride size）：1
        strides=[1,1,1,1]:前后两个1是固定的，的二个表示在height上的步长，的三个表示在width上的步长
    边距(padding size):0
        padding='SAME':
            'SAME':当下一步滑动后，像素点不够时自动填充
            'VALID':当下一步滑动后，像素点不够时将舍弃这步
            所以same的大于等于valid的输出

池化（max pooling）：
    2*2
"""

def weight_variable(shape):
    """
    创建权重矩阵，卷积核
    :param shape:矩阵的形状[filter_height, filter_width, in_channels, out_channels]
    :return:
    """
    initial = tf.truncated_normal(shape=shape, stddev=0.1) # 按照正态分布生成随机数
    return tf.Variable(initial)

def bias_variable(shape):
    """
    偏置项
    :param shape:
    :return:
    """
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """
    卷积
    :param x:输入数据
    :param W: 卷积核
    :return:
    """
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    """
    池化
    :param x: 输入数据
    :return:
    """
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

## 第一层  一个卷积+一个max pooling
W_conv1 = weight_variable([5,5,1,32]) # 32个5x5的卷积核输出32个图
b_conv1 = bias_variable([32])  # 每个卷积核对应的偏置量
# 为了使用这一层，我们把x变成一个4d的向量，第2、3维表示图的高和宽，最后一维表示图片的颜色通道数（因为是灰度图所以这里的通道数是1，如果是rgb彩色图，则为3）
x_image = tf.reshape(x, [-1,28,28,1])  # -1 表示会根据后三维自动生成
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # ReLU激活函数 f(x)=max(0,x)
h_pool1 = max_pool_2x2(h_conv1)

## 第二层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

## 密集连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])  # 1024个神经元的全连接 图片尺寸减小到7*7
b_fc1 = bias_variable([1024])  # 行形式
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  # 把前两层输出的结果拉成一维的向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

## dropout
"""
为了减小过拟合，一般用在全连接层
Dropout就是在不同的训练过程中随机扔掉一部分神经元，也就是让某个神经元的激活值以一定的概率p，
让其停止工作，这次训练过程中不更新权值也不参加神经网络的计算。但是它的权重得保留下来（暂时不更新），
下次个能会在更新
train的时候才起作用，test时不应让dropout起作用
"""
keep_prob = tf.placeholder(tf.float32)    # 每个元素被留下来的概率
h_fcl_drop = tf.nn.dropout(h_fc1, keep_prob)   #用在全连接层

## 输出层 使用softmax
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fcl_drop, W_fc2) + b_fc2)

## 评估模型
cross_entropy = -tf.reduce_sum(y*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# start = time.clock() # cpu时间
start = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%50 == 0:
            """
            （1）交互式使用  sess = tf.InteractiveSession()
            使用InteractiveSession代替Session，使用Tensor.eval()和Operation.run()方法代替Session.run()
            （2）或者在一个已经启动的绘画的图中：
                    with tf.Session() as sess:
            
            """
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y: batch[1], keep_prob: 1.0})
            print "step %d, training accuracy %g"%(i, train_accuracy)
        train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
    print "test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
# print('adam用时：{0}'.format(time.clock()-start))
print('adam用时：{0}'.format(time.time()-start))# 大概一个小时
"""
官网说这个的准确率99.2%

"""
