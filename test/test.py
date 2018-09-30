#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/10 10:04
# @Author  : Ivan Duan
# @Site    : 
# @Software: PyCharm
# import tensorflow as tf
# a = tf.constant([[0.91, 0.6, 0.95, 0.01], [1., 1., 1., 1.]])
# b = tf.greater(a, 0.9)
# c = tf.cast(b, tf.float32)
# with tf.Session() as sess:
#     print(sess.run(b))

# -------------------------
# a = tf.constant([1., 1., 0., 0., 0.])
# b = tf.constant([1., 1., 0., 1., 0.])
# c = tf.equal(a, b)
# d = tf.cast(c, tf.float32)
# e = tf.equal(tf.reduce_sum(d), tf.reduce_sum(tf.ones_like(d)))
# with tf.Session() as sess:
#     print(sess.run(c))
#     print(sess.run(d))
#     print(sess.run(e))

# -----------------------
# a = '电子科技大学'
# b = list(a)
# print(b)
# print(type(b))

# ------------------
import numpy as np
from numpy import array
# a = array([[1,2,3],
#      [4,5,6]])
# b = array([[1,2,3],
#      [4,5,6]])
# c = a + b
# c = list(c)
# print(type(c))

# ------------------
# a = array([[1, 2, 3, 4, 5],
#           [6, 7, 8, 9, 0]
#           ])
#
# print(a[0].shape)
# -----------------------
# arr1 = array([[1,2,3], [4,5,6]])
# arr2 = array([[7,8,9], [10,11,12]])
# data = np.concatenate([arr1, arr2], axis=1)
# print(data)

# ---------------------
# from scipy.sparse import *
# row = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
# col = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
# data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
# a = csr_matrix((data, (row, col)), shape=(4, 3))
# print(a.toarray())
# print(a[:2].toarray())
# x1 = a[2,:]
# x2 = a[2:]
# print(x1.shape)
# print(x1)
# x = vstack((x1, x2), format='csr')
# print(type(x))
# print(x.toarray())

# ------------------------
a = [[0, 1],
     [2, 3],
     ]
b = [[4, 5],
     [6, 7]]
c = np.concatenate([a, b], axis=0)
print(c)
print(type(c))