# !/usr/bin/env python
# -*- coding: utf8 -*-
# author:xuehansheng(xhs1892@gmail.com)

import numpy as np
from sklearn import preprocessing

yeast_nums = 6400
dim_nums = 500
yeast_nums_filter = 4443 #4061 #4428 #4443

human_nums = 18362
human_dims = 800
human_nums_filter = 3219#MF:3219#3237#2408#CC:1408#2284#2890 #BP:3541 #4254 #3984

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
	#14: m-AUPR = 0.739  M-AUPR = 0.605  F1 = 0.595  Acc = 0.814
	#    m-AUPR = 0.660  M-AUPR = 0.478  F1 = 0.548  Acc = 0.769
	# coexpression_emb = read_embeding('emb/yeast_coexpression_emb_semiAE_01_100_100000_5_SGD_rwr_pt_099.txt')
	# cooccurence_emb = read_embeding('emb/yeast_cooccurence_emb_semiAE_01_100_100000_5_SGD_rwr_pt_099.txt')
	# database_emb = read_embeding('emb/yeast_database_emb_semiAE_01_100_100000_5_SGD_rwr_pt_099.txt')
	# experimental_emb = read_embeding('emb/yeast_experimental_emb_semiAE_01_100_100000_5_SGD_rwr_pt_099.txt')
	# fusion_emb = read_embeding('emb/yeast_fusion_emb_semiAE_01_100_100000_5_SGD_rwr_pt_099.txt')
	# neighborhood_emb = read_embeding('emb/yeast_neighborhood_emb_semiAE_01_100_100000_5_SGD_rwr_pt_099.txt')

	#13: m-AUPR = 0.743  M-AUPR = 0.608  F1 = 0.594  Acc = 0.816
	#    m-AUPR = 0.655  M-AUPR = 0.481  F1 = 0.546  Acc = 0.766
	# coexpression_emb = read_embeding('emb/yeast_coexpression_emb_AE_01_128_100000_5_SGD_rwr_pt.txt')
	# cooccurence_emb = read_embeding('emb/yeast_cooccurence_emb_AE_01_128_100000_5_SGD_rwr_pt.txt')
	# database_emb = read_embeding('emb/yeast_database_emb_AE_01_128_100000_5_SGD_rwr_pt.txt')
	# experimental_emb = read_embeding('emb/yeast_experimental_emb_AE_01_128_100000_5_SGD_rwr_pt.txt')
	# fusion_emb = read_embeding('emb/yeast_fusion_emb_AE_01_128_100000_5_SGD_rwr_pt.txt')
	# neighborhood_emb = read_embeding('emb/yeast_neighborhood_emb_AE_01_128_100000_5_SGD_rwr_pt.txt')

	#10: m-AUPR = 0.738  M-AUPR = 0.606  F1 = 0.596  Acc = 0.820
	#    m-AUPR = 0.658  M-AUPR = 0.482  F1 = 0.544  Acc = 0.760
	# coexpression_emb = read_embeding('emb/yeast_coexpression_emb_AE_01_128_200000_5_SGD_rwr_pt_.txt')
	# cooccurence_emb = read_embeding('emb/yeast_cooccurence_emb_AE_01_128_200000_5_SGD_rwr_pt_.txt')
	# database_emb = read_embeding('emb/yeast_database_emb_AE_01_128_200000_5_SGD_rwr_pt_.txt')
	# experimental_emb = read_embeding('emb/yeast_experimental_emb_AE_01_128_200000_5_SGD_rwr_pt_.txt')
	# fusion_emb = read_embeding('emb/yeast_fusion_emb_AE_01_128_200000_5_SGD_rwr_pt_.txt')
	# neighborhood_emb = read_embeding('emb/yeast_neighborhood_emb_AE_01_128_200000_5_SGD_rwr_pt_.txt')

	#12: m-AUPR = 0.741  M-AUPR = 0.609  F1 = 0.596  Acc = 0.818
	#    m-AUPR = 0.650  M-AUPR = 0.473  F1 = 0.540  Acc = 0.757
	# coexpression_emb = read_embeding('emb/yeast_coexpression_emb_semiAE_01_128_200000_5_SGD_rwr_pt_.txt')
	# cooccurence_emb = read_embeding('emb/yeast_cooccurence_emb_semiAE_01_128_200000_5_SGD_rwr_pt_.txt')
	# database_emb = read_embeding('emb/yeast_database_emb_semiAE_01_128_200000_5_SGD_rwr_pt_.txt')
	# experimental_emb = read_embeding('emb/yeast_experimental_emb_semiAE_01_128_200000_5_SGD_rwr_pt_.txt')
	# fusion_emb = read_embeding('emb/yeast_fusion_emb_semiAE_01_128_200000_5_SGD_rwr_pt_.txt')
	# neighborhood_emb = read_embeding('emb/yeast_neighborhood_emb_semiAE_01_128_200000_5_SGD_rwr_pt_.txt')

	#11: m-AUPR = 0.746  M-AUPR = 0.612  F1 = 0.597  Acc = 0.822
	# coexpression_emb = read_embeding('emb/yeast_coexpression_emb_semiAE_01_128_200000_5_SGD_rwr_pt_999.txt')
	# cooccurence_emb = read_embeding('emb/yeast_cooccurence_emb_semiAE_01_128_200000_5_SGD_rwr_pt_999.txt')
	# database_emb = read_embeding('emb/yeast_database_emb_semiAE_01_128_200000_5_SGD_rwr_pt_999.txt')
	# experimental_emb = read_embeding('emb/yeast_experimental_emb_semiAE_01_128_200000_5_SGD_rwr_pt_999.txt')
	# fusion_emb = read_embeding('emb/yeast_fusion_emb_semiAE_01_128_200000_5_SGD_rwr_pt_999.txt')
	# neighborhood_emb = read_embeding('emb/yeast_neighborhood_emb_semiAE_01_128_200000_5_SGD_rwr_pt_999.txt')

	coexpression_emb = read_embeding('emb/layers/yeast_coexpression_emb_semiAE_01_128_200000_5_SGD_rwr_pt_999.txt')
	cooccurence_emb = read_embeding('emb/layers/yeast_cooccurence_emb_semiAE_01_128_200000_5_SGD_rwr_pt_999.txt')
	database_emb = read_embeding('emb/layers/yeast_database_emb_semiAE_01_128_200000_5_SGD_rwr_pt_999.txt')
	experimental_emb = read_embeding('emb/layers/yeast_experimental_emb_semiAE_01_128_200000_5_SGD_rwr_pt_999.txt')
	fusion_emb = read_embeding('emb/layers/yeast_fusion_emb_semiAE_01_128_200000_5_SGD_rwr_pt_999.txt')
	neighborhood_emb = read_embeding('emb/layers/yeast_neighborhood_emb_semiAE_01_128_200000_5_SGD_rwr_pt_999.txt')

	merge_emb = np.zeros((6, dim_nums, yeast_nums))
	merge_emb[0] = scale_emb(coexpression_emb)
	merge_emb[1] = scale_emb(cooccurence_emb)
	merge_emb[2] = scale_emb(database_emb)
	merge_emb[3] = scale_emb(experimental_emb)
	merge_emb[4] = scale_emb(fusion_emb)
	merge_emb[5] = scale_emb(neighborhood_emb)

	return merge_emb

def load_emb_human():
	# coexpression_emb = read_embeding_human('emb/human_coexpression_emb_semiAE_01_500_10000_5_Adam_pt_099.txt')
	# cooccurence_emb = read_embeding_human('emb/human_cooccurence_emb_semiAE_01_500_10000_5_Adam_pt_099.txt')
	# database_emb = read_embeding_human('emb/human_database_emb_semiAE_01_500_10000_5_Adam_pt_099.txt')
	# experimental_emb = read_embeding_human('emb/human_experimental_emb_semiAE_01_500_10000_5_Adam_pt_099.txt')
	# fusion_emb = read_embeding_human('emb/human_fusion_emb_semiAE_01_500_10000_5_Adam_pt_099.txt')
	# neighborhood_emb = read_embeding_human('emb/human_neighborhood_emb_semiAE_01_500_10000_5_Adam_pt_099.txt')

	coexpression_emb = read_embeding_human('emb/human_coexpression_emb_AE_01_128_50000_5_Adam_pt.txt')
	cooccurence_emb = read_embeding_human('emb/human_cooccurence_emb_AE_01_128_50000_5_Adam_pt.txt')
	database_emb = read_embeding_human('emb/human_database_emb_AE_01_128_50000_5_Adam_pt.txt')
	experimental_emb = read_embeding_human('emb/human_experimental_emb_AE_01_128_50000_5_Adam_pt.txt')
	fusion_emb = read_embeding_human('emb/human_fusion_emb_AE_01_128_50000_5_Adam_pt.txt')
	neighborhood_emb = read_embeding_human('emb/human_neighborhood_emb_AE_01_128_50000_5_Adam_pt.txt')

	# filepath = '/media/userdisk1/jjpeng/xuehansheng/emb/human/'
	# emb_path = 'emb_semiAE_01_500_10000_5_Adam_pt_099_MC.txt'	# 0.504 0.456 0.376 0.545
	# emb_path = 'emb_semiAE_05_128_50000_5_SGD_pt.txt'		# 0.482 0.432 0.359 0.516
	# emb_path = 'emb_semiAE_075_128_100000_5_SGD_pt_rwr.txt'	# 0.260 0.208 0.217 0.267
	# emb_path = 'emb_AE_05_128_50000_5_SGD_pt_rwr.txt'		# 0.255 0.204 0.215 0.259
	# coexpression_emb = read_embeding_human(filepath+'human_neighborhood_'+emb_path)
	# cooccurence_emb = read_embeding_human(filepath+'human_fusion_'+emb_path)
	# database_emb = read_embeding_human(filepath+'human_experimental_'+emb_path)
	# experimental_emb = read_embeding_human(filepath+'human_database_'+emb_path)
	# fusion_emb = read_embeding_human(filepath+'human_cooccurence_'+emb_path)
	# neighborhood_emb = read_embeding_human(filepath+'human_coexpression_'+emb_path)

	# coexpression_emb = read_embeding_human('/home/jjpeng/xuehansheng/Old/human_coexpression_emb_semiAE_01_128_50000_5_Adam_pt.txt')
	# cooccurence_emb = read_embeding_human('/home/jjpeng/xuehansheng/Old/human_cooccurence_emb_semiAE_01_128_50000_5_Adam_pt.txt')
	# experimental_emb = read_embeding_human('/home/jjpeng/xuehansheng/Old/human_experimental_emb_semiAE_01_128_50000_5_Adam_pt.txt')

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