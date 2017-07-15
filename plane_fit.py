# coding=utf-8
import tensorflow as tf
import numpy as np

# 使用numpy生成假数据，共100个点
x_data = np.float32(np.random.rand(2, 100))  # 2x 100的数组
# print x_data
y_data = np.dot([0.1, 0.2], x_data) + 0.3
# print y_data

# 构造一个线性函数
W = tf.Variable(tf.random_normal([1, 2]))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
# init = tf.initialize_all_variables()  #  过时
init = tf.global_variables_initializer()

# 启动图
with tf.Session() as sess:
    sess.run(init)
    for step in range(200):
        sess.run(train)
        print(step,sess.run(W),sess.run(b))


