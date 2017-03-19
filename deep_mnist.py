from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784]) #Input Layer
y_ = tf.placeholder(tf.float32, shape=[None, 10]) #Target Output

W = tf.Variable(tf.zeros([784,10])) #Weights
b = tf.Variable(tf.zeros([10])) #Offsets

