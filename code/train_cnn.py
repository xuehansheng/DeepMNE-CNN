# !/usr/bin/env python
# -*- coding: utf8 -*-
# author:xuehansheng(xhs1892@gmail.com)

import os
import time
import datetime
import numpy as np
import tensorflow as tf
from utils import *
from load_mips import load_mips
from load_emb import load_emb, filter_emb
from models import BioCNN
from tensorflow.contrib import learn
from sklearn import metrics
from validation import evaluate_performance
from sklearn.model_selection import ShuffleSplit, KFold

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
log_device_placement = True

# Set random seed
seed = 2
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('org', 'Yeast', 'Dataset string')  # 'yeast', 'human'
flags.DEFINE_string('onttype', 'level1', 'which type of annotations to use options')
flags.DEFINE_float('percent_split', 0.2, 'Percentage of Spliting dataset')
flags.DEFINE_integer('num_epochs', 300, 'Number of Epoches')
flags.DEFINE_float('learning_rate', 5e-3, 'Initial learning rate')
flags.DEFINE_integer('batch_size', 64, 'Initial batch size')
flags.DEFINE_list('filter_sizes', [7, 5], 'Size of each filter')
flags.DEFINE_integer('filter_nums', 256, 'Number of filter')
flags.DEFINE_float('l2_reg_lambda',0.1, 'Number of l2_reg_lambda')
flags.DEFINE_string('optimizer', 'Adade', 'Optimizer')
flags.DEFINE_float('Dropout', 0.5, 'Number of Dropout')
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")


# Load embeding and annotaion
print("Loading data and annotation...")
embeding = load_emb(FLAGS.org) # (6,500,6400)
annotation = load_mips(FLAGS.onttype) # (17,6400)
# print(embeding.shape, annotation.shape)
# print annotation

# Filter genes with no annotations
print("Filtering data and annotation...")
emb, anno = filter_emb(embeding, annotation) # (6,4443,500)(4443,17) / (6,4428,500)(4428,74)
print(emb.shape, anno.shape)
# print anno.sum(axis=0)
# emb = np.transpose(emb,(1,2,0))
# print(emb.shape)

# Randomly shuffle data
shuffle_indices = np.random.permutation(np.arange(len(anno)))
emb_shuffled = emb[:,shuffle_indices,:] # (6,4443,500)
anno_shuffled = anno[shuffle_indices] # (4443,17)

# Split train/test set
test_sample_percentage = FLAGS.percent_split
test_sample_index = -1 * int(test_sample_percentage * float(len(anno)))
emb_train, emb_test = emb_shuffled[:,:test_sample_index,:], emb_shuffled[:,test_sample_index:,:]#(6,3555,500),(6,888,500)
anno_train, anno_test = anno_shuffled[:test_sample_index], anno_shuffled[test_sample_index:]#(3555,17),(888,17)

# # Split train/dev set
dev_sample_percentage = FLAGS.percent_split
dev_sample_index = -1 * int(dev_sample_percentage * float(len(anno_train)))
emb_train, emb_dev = emb_train[:,:dev_sample_index,:], emb_train[:,dev_sample_index:,:] #(6,3195,500),(6,355,500)
anno_train, anno_dev = anno_train[:dev_sample_index], anno_train[dev_sample_index:] #(3195,17),(355,17)
# print(emb_train.shape, emb_dev.shape, emb_test.shape) # (6, 3600, 500) (6, 399, 500) (6, 444, 500)
# print(anno_train.shape, anno_dev.shape, anno_test.shape) # (3600, 17) (399, 17) (444, 17)

# Training
with tf.Graph().as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement,
		device_count={'gpu':0})
	session_conf.gpu_options.allow_growth = True
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		cnn = BioCNN(
			num_input=emb_train.shape[1],
			num_classes=anno_train.shape[1],
			embeding_size=emb_train.shape[2],
			feature_size=emb_train.shape[0],
			filter_sizes=FLAGS.filter_sizes,
			num_filters=FLAGS.filter_nums,
			l2_reg_lambda=FLAGS.l2_reg_lambda)

		# Define training procedure
		global_step = tf.Variable(0, name="global_step", trainable=False)
		if FLAGS.optimizer == 'SGD':
			optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate) #1e-3
		elif FLAGS.optimizer == 'Momentum':
			optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.9)
		elif FLAGS.optimizer == 'RMSProp':
			optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate, 0.9)
		elif FLAGS.optimizer == 'Adam':
			optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate) #1e-6
		elif FLAGS.optimizer == 'Adade':
			optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate) #2e-4
		grads_and_vars = optimizer.compute_gradients(cnn.loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		# Output directory for models and summaries
		timestamp = str(int(time.time()))
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
		print("Writing to {}\n".format(out_dir))

		# Summaries for loss and accuracy
		loss_summary = tf.summary.scalar("loss", cnn.loss)
		acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

		# Train Summaries
		train_summary_op = tf.summary.merge([loss_summary, acc_summary])
		train_summary_dir = os.path.join(out_dir, "summaries", "train")
		train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

		# Dev summaries
		dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
		dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
		dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

		# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
		checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

		# Initialize all varibales
		sess.run(tf.global_variables_initializer())

		def train_step(x_batch, y_batch):
			feed_dict = {
			cnn.input_x: x_batch,
			cnn.input_y: y_batch,
			cnn.dropout_keep_prob: FLAGS.Dropout
			}
			_, step, summaries, loss, accuracy = sess.run(
				[train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
			if step % 50 == 0 or step == 1:
				print("step {}: loss {:g}, acc {:g}".format(step, loss, accuracy))
			train_summary_writer.add_summary(summaries, step)

		def dev_step(x_batch, y_batch, writer=None):
			feed_dict = {
			cnn.input_x: x_batch,
			cnn.input_y: y_batch,
			cnn.dropout_keep_prob: FLAGS.Dropout
			}
			step, summaries, loss, accuracy = sess.run(
				[global_step, dev_summary_op, cnn.loss, cnn.accuracy], feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print("{}: Step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
			if writer:
				writer.add_summary(summaries, step)
			return accuracy

		# Generate batches
		emb_train_ = np.transpose(emb_train,(1,2,0))#(None, 500, 6)
		emb_dev_ = np.transpose(emb_dev,(1,2,0))
		batches = batch_iter(list(zip(emb_train_, anno_train)), FLAGS.batch_size, FLAGS.num_epochs)
		# max_acc = 0
		for batch in batches:
		# for loop in xrange(num_epochs):
			x_batch, y_batch = zip(*batch) #(64,500,6) (64,17)
			train_step(x_batch, y_batch)
			# train_step(emb_train_, anno_train)
			current_step = tf.train.global_step(sess, global_step)
			if current_step % 500 == 0:
				print("\nEvaluation:")
				val_acc = dev_step(emb_dev_, anno_dev, writer=dev_summary_writer)
				print("")
				# if val_acc > max_acc:
					# max_acc = val_acc
				path = saver.save(sess, checkpoint_prefix, global_step=current_step)
				print("Saved model checkpoint to {}\n".format(path))

# Evaluation
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement)
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		# Load the saved meta graph and restore variables
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess, checkpoint_file)
		input_x = graph.get_operation_by_name("input_x").outputs[0]
		# input_y = graph.get_operation_by_name("input_y").outputs[0]
		dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
		# Tensors we want to evaluate
		predictions = graph.get_operation_by_name("output/predictions").outputs[0]
		scores = graph.get_operation_by_name("output/scores").outputs[0]

		emb_test_ = np.transpose(emb_test,(1,2,0))
		pre_labels, pre_scores = sess.run([predictions, scores], {input_x: emb_test_, dropout_keep_prob: 1.0})
		y_test = anno_test

print("\nEvaluation:")
print
pref = evaluate_performance(y_test, pre_scores, pre_labels)
print("micro-AUPRC: {:g}, macro-AUPRC: {:g}, F1: {:g}, Acc: {:g}".format(pref['m-aupr'], pref['M-aupr'], pref['F1'], pref['acc']))
print
print('Parameters:',FLAGS.optimizer, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.num_epochs, FLAGS.Dropout, FLAGS.l2_reg_lambda)
