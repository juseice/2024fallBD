import tensorflow as tf
tf.debugging.set_log_device_placement(True)

# Create a simple computation
a = tf.constant([[1.0, 2.0, 3.0]])
b = tf.constant([[4.0], [5.0], [6.0]])
c = tf.matmul(a, b)
print(c)