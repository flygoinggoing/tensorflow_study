#coding=utf-8
import tensorflow as tf

"""
自动下载并导入MNIST数据集  每张图28*28 
特征表示在导入时已经处理好
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 标签采用one_hot表示


"""
softmax regression
y = softmax(Wx + b)

"""
x = tf.placeholder(tf.float32, [None, 784])   # 把图展开成一维向量
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax模型
y_predict = tf.nn.softmax(tf.add(tf.matmul(x, W), b))  # y_predict是行形式的  xW也使得结果是一行一个比较好好看

# 交叉熵
cross_entropy = -tf.reduce_sum(y * tf.log(y_predict))  # 计算y对应的每一个元素，然后默认按行做和做和

# 最小化交叉熵梯度下降
train = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化op
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1, 1001):

        # 训练
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x:batch_xs, y:batch_ys})

        if step % 10 == 0:
            # 模型评估
            # tf.argmax返回对象在某一维上的数据最大值的索引
            # tf.equal是否相等（索引值一样时相等）  返回True或False
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_predict, 1))  # 1： y按列做比较
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # cast 变量的类型转换  reduce_mean:做和取平均
            print 'step:',step,sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})





