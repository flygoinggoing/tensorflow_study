#coding=utf-8
import tensorflow as tf
import numpy as np

"""
transpose：
a_arr中的8的下标为（2,1,2）三维对应0，1，2
经过perm=[0,2,1]==>下标变为(2,2,1)

"""
a = [[[1,2,3],[4,5,6]],[[7,8,9],[11,12,13]]]
print('a:')
print (a)


a_arr = np.array(a)
print('a_arr:')
print(a_arr)

with tf.Session() as session:
    print(session.run(tf.transpose(a_arr,perm=[0,2,1],name='transpose')))
