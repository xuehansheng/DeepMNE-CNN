# !/usr/bin/env python
# -*- coding: utf8 -*-
# author:xuehansheng(xhs1892@gmail.com)

import numpy as np
import scipy.io as sio
import tensorflow as tf
from utils import *

from math import sqrt
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import array_ops

class SemiAutoEncoder:

	def __init__(self, input_dim, hidden_dim, learning_rate, batch_size, num_steps, gamma, alpha):
		self.num_input = input_dim
		self.num_hidden = hidden_dim
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.num_steps = num_steps
		self.display_step = 1000
		self.examples_to_show = 10
		self.epsilon = 1e-3
		self.batch_size = batch_size
		self.gamma = gamma
		self.alpha = alpha

		self.weights = {
    		'encoder': tf.Variable(tf.random_normal([self.num_input, self.num_hidden])),
    		'decoder': tf.Variable(tf.random_normal([self.num_hidden, self.num_input])),
		}
		self.biases = {
    		'encoder': tf.Variable(tf.random_normal([self.num_hidden])),
    		'decoder': tf.Variable(tf.random_normal([self.num_input])),
		}

	# Building the encoder
	def encoder(self, x):
	    # Encoder Hidden layer with sigmoid activation
	    z = tf.add(tf.matmul(x, self.weights['encoder']), self.biases['encoder'])
	    batch_mean, batch_var = tf.nn.moments(z, [0])
	    scale = tf.Variable(tf.ones([self.num_hidden]))
	    beta = tf.Variable(tf.zeros([self.num_hidden]))
	    bn = tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, self.epsilon)   
	    layer = tf.nn.sigmoid(bn)

	    return layer

	# Building the decoder
	def decoder(self, x):
	    # Decoder Hidden layer with sigmoid activation
	    z = tf.add(tf.matmul(x, self.weights['decoder']), self.biases['decoder'])
	    batch_mean, batch_var = tf.nn.moments(z, [0])
	    scale = tf.Variable(tf.ones([self.num_input]))
	    beta = tf.Variable(tf.zeros([self.num_input]))
	    bn = tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, self.epsilon)
	    layer = tf.nn.sigmoid(bn)

	    return layer

	def get_constraints_loss(self, hidden, constraints_batch):
		D = tf.diag(tf.reduce_sum(constraints_batch,1))
		L = D - constraints_batch
		loss = 2*tf.trace(tf.matmul(tf.matmul(tf.transpose(hidden),L),hidden))
		return loss/(self.batch_size*self.batch_size)
		# return loss

	def train(self, data, mustlink, cannotlink, opt):
		# data_iter = Dataset(data)
		data_iter = SemiDataset(data, mustlink, cannotlink)

		X = tf.placeholder("float", [None, self.num_input])
		# Data = tf.placeholder('float', [None, self.num_input])
		# X = tf.placeholder("float", [None, None])
		# Construct model
		encoder_op = self.encoder(X)
		decoder_op = self.decoder(encoder_op)
		# Prediction
		y_pred = decoder_op
		y_true = X
		# Define loss and optimizer, minimize the squared error
		loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

		constraints = tf.placeholder("float", [None, self.batch_size])
		constraints_ = tf.placeholder("float", [None, self.batch_size])
		loss_ml = self.get_constraints_loss(encoder_op, constraints)
		loss_cl = self.get_constraints_loss(encoder_op, constraints_)
		# gamma alpha
		loss_value = loss + self.gamma*loss_ml - self.alpha*loss_cl
 
		if opt == 'SGD':
			optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
		elif opt == 'Adam':
			optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
		elif opt == 'Adade':
			optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(loss)
		elif opt == 'RSMP':
			optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss)
		elif opt == 'Momentum':
			optimizer = tf.train.MomentumOptimizer(self.learning_rate).minimize(loss)

		# Initialize the variables (i.e. assign their default value)
		init = tf.global_variables_initializer()
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True
		with tf.Session(config=config) as sess:
		    # Run the initializer
			sess.run(init)
			# Training
			for i in range(1, self.num_steps + 1):
				batch_x, batch_m, batch_c = data_iter.next_batch(self.batch_size)
				# print batch_x,batch_m
				_, l = sess.run([optimizer, loss_value], feed_dict={X: batch_x, constraints: batch_m, constraints_: batch_c})
				if i % self.display_step == 0 or i == 1:
					print('Step %i: Loss: %f' % (i, l))
				hidden_value = sess.run([encoder_op], feed_dict={X: data})

			return hidden_value[0]


class BioCNN(object):

	def __init__(self, num_input, num_classes, embeding_size, feature_size, filter_sizes, num_filters, l2_reg_lambda):
		# num_input:6400, num_classes=17, embeding_size:500, feature_size=6
		# Placeholders for input, output and dropout
		self.input_x = tf.placeholder(tf.float32, [None, embeding_size, feature_size], name="input_x")
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		# keeping track of l2 regularization loss (optimal)
		l2_loss = tf.constant(0.0)
		self.input_xx = tf.expand_dims(self.input_x, -1)
		# print self.input_xx.shape #(?, 500, 6, 1)
		print(self.input_xx.shape) #(?, 500, 6, 1)

		# Create a convolution + maxpool layer for each filter size
		# num_conv = 3
		h_outputs = []
		for filter_size in filter_sizes:
		# with tf.name_scope("conv-maxpool-layer"):
			# Conv+ReLu+BN+Pool - 1
			filter_width = self.input_xx.get_shape()[2].value
			filter_shape = [filter_size, filter_width, 1, num_filters]
			# print filter_shape
			W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
			b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
			conv = tf.nn.conv2d(self.input_xx, W, strides=[1,1,1,1], padding="VALID", name="conv")
			h = tf.nn.bias_add(conv, b)
			h = self.batch_normalization_layer(h)
			h = tf.nn.relu(h, name="relu")
			print(h.shape)

			h_outputs.append(h)

		outputs = tf.concat(h_outputs, 1)
		# print outputs.shape
		print(outputs.shape)
		outputs = tf.nn.max_pool(outputs, ksize=[1,2,1,1], strides=[1,2,1,1], padding="VALID", name="pool")
		# print outputs.shape
		print(outputs.shape)

		with tf.name_scope("ReshapeLayer"):
			vec_dim = outputs.get_shape()[1].value * outputs.get_shape()[2].value * outputs.get_shape()[3].value
			x = tf.reshape(outputs, [-1, vec_dim])
		# print x.shape
		print(x.shape)

		with tf.name_scope("LinearLayer"):
			stdv = 1 / sqrt(vec_dim)
			W = tf.Variable(tf.random_uniform([vec_dim, num_filters*len(filter_sizes)], minval=-stdv, maxval=stdv), dtype='float32', name='W')
			b = tf.Variable(tf.random_uniform(shape=[num_filters*len(filter_sizes)], minval=-stdv, maxval=stdv), dtype='float32', name = 'b')
			x = tf.nn.xw_plus_b(x,W,b)
			# x = self.batch_normalization_layer(x)
			x = tf.nn.relu(x, name="relu")

		with tf.name_scope("DropoutLayer"):
			x = tf.nn.dropout(x, self.dropout_keep_prob)
		# print x.shape
		print(x.shape)

		# Final (unnormalized) scores and predictions
		with tf.name_scope("output"):
			W = tf.get_variable("W", shape=[num_filters*len(filter_sizes), num_classes], initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
			# stdv = np.sqrt(3.0/num_filters)
			# W = tf.Variable(tf.random_uniform([num_filters, num_classes], minval=-stdv, maxval=stdv), dtype='float32', name='W')
			# b = tf.Variable(tf.random_uniform(shape=[num_classes], minval=-stdv, maxval=stdv), dtype='float32', name = 'b')
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.scores = tf.nn.xw_plus_b(x, W, b, name="scores")
			self.predictions = tf.argmax(self.scores, 1, name="predictions")

		# Calculate mean cross-entropy loss
		with tf.name_scope("loss"):
			# losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
			losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

		# Accuracy
		with tf.name_scope("accuracy"):
			# correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			# print self.scores.shape, self.input_y.shape
			true_pred_idx = tf.argmax(tf.multiply(self.scores, self.input_y), 1)
			correct_predictions = tf.equal(self.predictions, true_pred_idx)
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

	def batch_normalization_layer(self, inputs, isTrain=True):
		# TODO: implemented the batch normalization func and applied it on conv and fully-connected layers
		# hint: you can add extra parameters (e.g., shape) if necessary
		EPSILON = 0.01
		CHANNEL = inputs.shape[3]
		MEANDECAY = 0.99

		ave_mean = tf.Variable(tf.zeros(shape = [CHANNEL]), trainable = False)
		ave_var = tf.Variable(tf.zeros(shape = [CHANNEL]), trainable = False)
		mean, var = tf.nn.moments(inputs, axes=[0,1,2], keep_dims=False)

		update_mean_op = moving_averages.assign_moving_average(ave_mean, mean, MEANDECAY)
		update_var_op = moving_averages.assign_moving_average(ave_var, var, MEANDECAY)

		tf.add_to_collection("update_op", update_mean_op)
		tf.add_to_collection("update_op", update_var_op)

		scale = tf.Variable(tf.constant(1.0, shape=mean.shape))
		offset = tf.Variable(tf.constant(0.0, shape=mean.shape))

		if isTrain:
			inputs = tf.nn.batch_normalization(inputs, mean = mean, variance = var, offset = offset, scale = scale, variance_epsilon = EPSILON)
		else:
			inputs = tf.nn.batch_normalization(inputs, mean = ave_mean, variance = ave_var, offset = offset, scale = scale, variance_epsilon = EPSILON)

		return inputs
			