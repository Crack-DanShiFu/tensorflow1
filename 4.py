import tensorflow as tf

state = tf.Variable(0, name="state")
one = tf.constant(1)
new_Value = tf.add(state, one)
update = tf.assign(state, new_Value)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(10):
        sess.run(update)
        print(sess.run(state))
