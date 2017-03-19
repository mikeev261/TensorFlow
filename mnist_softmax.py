#Tensorflow Tutorial: MNIST for ML Beginners: https://www.tensorflow.org/get_started/mnist/beginners
#Relevant links: 
#Visualizing MNIST: An Exploration of Dimensionality Reduction: http://colah.github.io/posts/2014-10-Visualizing-MNIST/
#THE MNIST DATABASE: http://yann.lecun.com/exdb/mnist/
#Cross Entropy / Visual Information Theory: http://colah.github.io/posts/2015-09-Visual-Information/
#Backprop: http://colah.github.io/posts/2015-08-Backprop/
#List of Tensorflow Optimizers: https://www.tensorflow.org/api_guides/python/train#optimizers
#Gradient Descent: https://en.wikipedia.org/wiki/Gradient_descent
#What is the class of this image ? Discover the current state of the art in objects classification.http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html

import tensorflow as tf
#Python Profilers: https://docs.python.org/3/library/profile.html
import cProfile #C-version (low overhead) of the python profiler (for benchmarking)
import re #Regular Expressions
import pstats #for exporting cProfile.run into a readable format

def mnist_softmax():

	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


	x = tf.placeholder(tf.float32, [None, 784]) #MNIST input

	W = tf.Variable(tf.zeros([784, 10])) #weights (multiplied against our input x)
	b = tf.Variable(tf.zeros([10])) #bias (added later)

	y = tf.nn.softmax(tf.matmul(x, W)+ b) #our model!

	#Cross-Entropy (to measure inefficiency)
	y_ = tf.placeholder(tf.float32, [None, 10])
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	#tf.nn.softmax_cross_entropy_with_logits ?

	#Training
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	#Launch
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	for _ in range(1000):
	  batch_xs, batch_ys = mnist.train.next_batch(100)
	  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	#Evaluation
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	print("ACCURACY:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

cProfile.run('mnist_softmax()', 'benchstats')
bench = pstats.Stats('benchstats')
bench.strip_dirs()
bench.sort_stats('time', 'cumulative')
bench.print_stats(.01)