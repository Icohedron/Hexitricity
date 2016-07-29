import tensorflow as tf

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W, stride):
    return tf.Variable(tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME'))

class Network(object):

    x = tf.placeholder(tf.float32, [None, 13 * 13 + 1], name='Input')
    y_ = tf.placeholder(tf.float32, [None, 13 * 13], name='Output')

    def __init__(self):
        pass

    def feed_forward(self, x):
        pass

    def train(self):
        pass
