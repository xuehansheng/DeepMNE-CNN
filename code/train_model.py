# !/usr/bin/env python
# -*- coding: utf8 -*-
# author:xuehansheng(xhs1892@gmail.com)

import os
import datetime
import numpy as np
import tensorflow as tf
from models import *
from data_helpers import *
from validation import cross_validation

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
log_device_placement = True


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('org', 'yeast', 'Dataset string')  # 'yeast', 'human'
flags.DEFINE_integer('net_dims', 6400, 'Dimensional number of input networks')
flags.DEFINE_integer('net_nums', 6, 'Number of input networks')
# flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_list('learning_rate', [0.5, 0.05, 0.05, 0.01], 'Initial learning rate')
# flags.DEFINE_float('learning_rate', [0.1, 0.01, 0.01], 'Initial learning rate')
flags.DEFINE_integer('batch_size', 128, 'Initial batch size')
# flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_list('iter_num', [50000, 10000, 10000, 5000], 'Number of iterations to train')
# flags.DEFINE_integer('iter_num', [10000, 1000, 1000], 'Number of iterations to train')
flags.DEFINE_integer('layers_num', 1, 'Number of the whole model')
flags.DEFINE_list('hidden_dim', [500], 'Number of units in hidden layers')
# flags.DEFINE_integer('hidden_dim', [3200, 1600, 500], 'Number of units in hidden layers.')
# flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
# flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
# flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
# flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string('optimizer', 'SGD', 'Optimizer')
flags.DEFINE_float('gamma', 0.5, 'The weight of mustlink constraints')
flags.DEFINE_float('alpha', 0.5, 'The weight of cannotlink constraints')
flags.DEFINE_float('percent', 0.05, 'Pecentage of extracting constraints')


# load data
print ('Loading genes...')
yeast_genes = load_genes('yeast')
print ('Loading network fusions... ')
yeast_fusions = load_fusions('yeast')
# yeast_fusions = load_networks('yeast')

input_dim = FLAGS.net_dims

mustlinks = np.zeros((FLAGS.net_nums, FLAGS.net_dims, FLAGS.net_dims))
cannotlinks = np.zeros((FLAGS.net_nums, FLAGS.net_dims, FLAGS.net_dims))
constraints_ml = np.zeros((FLAGS.net_nums, FLAGS.net_dims, FLAGS.net_dims))
constraints_cl = np.zeros((FLAGS.net_nums, FLAGS.net_dims, FLAGS.net_dims))

for idx_layer in range(FLAGS.layers_num):
	emb = np.zeros((FLAGS.net_nums, FLAGS.net_dims, FLAGS.hidden_dim[idx_layer]))

	for idx_net in range(FLAGS.net_nums):

		semiAE = SemiAutoEncoder(input_dim, FLAGS.hidden_dim[idx_layer], FLAGS.learning_rate[idx_layer], FLAGS.batch_size, FLAGS.iter_num[idx_layer], FLAGS.gamma, FLAGS.alpha)
		emb[idx_net] = semiAE.train(yeast_fusions[idx_net], constraints_ml[idx_net], constraints_cl[idx_net], FLAGS.optimizer)

	if idx_layer != FLAGS.layers_num - 1:
		print ('Extracting constraints...')
		for idx_net in range(FLAGS.net_nums):
			print(idx_net)
			temp_mustlink, temp_cannotlink = extractConstraints(emb[idx_net])
			# temp_mustlink, temp_cannotlink = extractConstraints_top(emb[idx_net], FLAGS.percent)
			mustlinks[idx_net] = temp_mustlink
			cannotlinks[idx_net] = temp_cannotlink

		print ('Merging constraints...')
		for idx in range(FLAGS.net_nums):
			temp_mustlink = np.zeros((FLAGS.net_dims, FLAGS.net_dims))
			temp_cannotlink = np.zeros((FLAGS.net_dims, FLAGS.net_dims))
			for idxx in range(FLAGS.net_nums):
				if idxx != idx:
					temp_mustlink = temp_mustlink + mustlinks[idxx]
					temp_cannotlink = temp_cannotlink + cannotlinks[idxx]
			constraints_ml[idx] = np.floor(temp_mustlink / (FLAGS.net_nums - 1))
			constraints_cl[idx] = np.floor(temp_cannotlink / (FLAGS.net_nums - 1))
			print (len(constraints_ml[idx].nonzero()[0]) / 2, len(constraints_cl[idx].nonzero()[0]) / 2)
		
	input_dim = FLAGS.hidden_dim[idx_layer]
	yeast_fusions = emb

# output embedding
str_nets = ['coexpression', 'cooccurence', 'database', 'experimental', 'fusion', 'neighborhood']
for idxx in range(FLAGS.net_nums):
	temp_path = './emb/'+FLAGS.org+'_'+str_nets[idxx]+'_'+FLAGS.optimizer+'_'+str(FLAGS.learning_rate[0])+'_new.txt'
	write_encoded_file(emb[idxx], temp_path)

perf = cross_validation(emb, labels)

print ("Average (over trials) of DeepMNE: m-AUPR = %0.3f, M-AUPR = %0.3f, F1 = %0.3f, Acc = %0.3f" 
	% (np.mean(perf['pr_micro']), np.mean(perf['pr_macro']), np.mean(perf['fmax']), np.mean(perf['acc'])))
print


print (FLAGS.layers_num, FLAGS.optimizer, FLAGS.learning_rate, FLAGS.batch_size)