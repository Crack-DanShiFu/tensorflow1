import tensorflow as tf
# import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = cv2.imread('1.png')
# for i in range(100):
#     img[20 + i, 100] = (255, 0, 0)
#
# cv2.imshow('image', img)
# cv2.waitKey(0)

# data1 = tf.constant(2, dtype=tf.int32)
# data2 = tf.Variable(10)
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# print(sess.run(tf.add(data1, data2)))
# print(sess.run(tf.multiply(data1, data2)))
# print(sess.run(tf.subtract(data1, data2)))
# print(sess.run(tf.divide(data1, data2)))

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([2, 4, 8, 3, 5, 8, 2, 5, 7])
plt.bar(x, y, 0.2, alpha=1, color='b')
plt.plot(x, y, 'b')
plt.show()
