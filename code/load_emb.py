# !/usr/bin/env python
# -*- coding: utf8 -*-
# author:xuehansheng(xhs1892@gmail.com)

import numpy as np
from sklearn import preprocessing

yeast_nums = 6400
dim_nums = 500
yeast_nums_filter = 4443

human_nums = 18362
human_dims = 800
human_nums_filter = 3219

def read_embeding(filepath):
	embeding = np.zeros([yeast_nums, dim_nums])
	data = open(filepath, "r")
	network = [[v for v in line.rstrip('\n').split('\t')] for line in data]
	# print len(network),len(network[0])
	for i in range(len(network)):
		for j in range(len(network[0])-1):
			embeding[i][j] = float(network[i][j])
	return np.array(embeding)
	# return embeding

def read_embeding_human(filepath):
	embeding = np.zeros([human_nums, human_dims])
	data = open(filepath, "r")
	network = [[v for v in line.rstrip('\n').split('\t')] for line in data]
	# print len(network),len(network[0])
	for i in range(len(network)):
		for j in range(len(network[0])-1):
			embeding[i][j] = float(network[i][j])
	return np.array(embeding)

# scale feature
def scale_emb(data):
	min_max_scaler = preprocessing.MinMaxScaler()
	emb_scaled = min_max_scaler.fit_transform(data)
	return np.transpose(emb_scaled)

def load_emb(org):
	if org == 'Yeast':
		emb = load_emb_yeast()
	elif org == 'Human':
		emb = load_emb_human()
	return emb

def load_emb_yeast():

	coexpression_emb = read_embeding('emb/layers/yeast_coexpression_emb_semiAE.txt')
	cooccurence_emb = read_embeding('emb/layers/yeast_cooccurence_emb_semiAE.txt')
	database_emb = read_embeding('emb/layers/yeast_database_emb_semiAE.txt')
	experimental_emb = read_embeding('emb/layers/yeast_experimental_emb_semiAE.txt')
	fusion_emb = read_embeding('emb/layers/yeast_fusion_emb_semiAE.txt')
	neighborhood_emb = read_embeding('emb/layers/yeast_neighborhood_emb_semiAE.txt')

	merge_emb = np.zeros((6, dim_nums, yeast_nums))
	merge_emb[0] = scale_emb(coexpression_emb)
	merge_emb[1] = scale_emb(cooccurence_emb)
	merge_emb[2] = scale_emb(database_emb)
	merge_emb[3] = scale_emb(experimental_emb)
	merge_emb[4] = scale_emb(fusion_emb)
	merge_emb[5] = scale_emb(neighborhood_emb)

	return merge_emb

def load_emb_human():
	coexpression_emb = read_embeding_human('emb/human_coexpression_emb_semiAE.txt')
	cooccurence_emb = read_embeding_human('emb/human_cooccurence_emb_semiAE.txt')
	database_emb = read_embeding_human('emb/human_database_emb_semiAE.txt')
	experimental_emb = read_embeding_human('emb/human_experimental_emb_semiAE.txt')
	fusion_emb = read_embeding_human('emb/human_fusion_emb_semiAE.txt')
	neighborhood_emb = read_embeding_human('emb/human_neighborhood_emb_semiAE.txt')

	merge_emb = np.zeros((6, human_dims, human_nums))
	merge_emb[0] = scale_emb(coexpression_emb)
	merge_emb[1] = scale_emb(cooccurence_emb)
	merge_emb[2] = scale_emb(database_emb)
	merge_emb[3] = scale_emb(experimental_emb)
	merge_emb[4] = scale_emb(fusion_emb)
	merge_emb[5] = scale_emb(neighborhood_emb)

	return merge_emb

def filter_emb(emb, anno):
	sum_col = anno.sum(axis=0)
	filter_emb = np.zeros((6, yeast_nums_filter, dim_nums))
	for k in range(6):
		temp_emb = emb[k]
		emb_filter = []
		anno_filter = []
		for i in range(len(sum_col)):
			if sum_col[i] > 0.0:
				emb_filter.append(temp_emb[:,i])
				anno_filter.append(anno[:,i])
		filter_emb[k] = np.array(emb_filter)
	return filter_emb, np.array(anno_filter)

def filter_emb_human(emb, anno):
	sum_col = anno.sum(axis=0)
	filter_emb = np.zeros((6, human_nums_filter, human_dims))
	for k in range(6):
		temp_emb = emb[k]
		emb_filter = []
		anno_filter = []
		for i in range(len(sum_col)):
			if sum_col[i] > 0.0:
				emb_filter.append(temp_emb[:,i])
				anno_filter.append(anno[:,i])
		filter_emb[k] = np.array(emb_filter)
	return filter_emb, np.array(anno_filter)