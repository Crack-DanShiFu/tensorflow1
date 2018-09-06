import tensorflow as tf

m1 = tf.constant([[3, 3]])
m2 = tf.constant([[2], [2]])

product = tf.matmul(m1, m2)

# sess = tf.Session()
# result = sess.run(product)
# print(result)

with tf.Session() as sess:
    print(sess.run(product))
