# !/usr/bin/env python
# -*- coding: utf8 -*-
# author:xuehansheng(xhs1892@gmail.com)

import numpy as np
from sklearn import preprocessing

yeast_nums = 6400

def load_genes(org):
	if org == 'yeast':
		genes = readGenes('data/networks/yeast/yeast_string_genes.txt')
	elif org == 'human':
		genes = readGenes('data/networks/human/human_string_genes.txt')
	return genes

def load_networks(org):
	num_nets = 6
	str_nets = ['coexpression', 'cooccurence', 'database', 'experimental', 'fusion', 'neighborhood']
	adj_nets = np.zeros((num_nets, yeast_nums, yeast_nums))
	for idx in range(num_nets):
		path_net = 'data/networks/'+org +'/'+org+'_string_'+str_nets[idx]+'_adjacency.txt'
		adj_nets[idx] = readNetworks(path_net)
	return adj_nets

def load_fusions(org):
	num_fusions = 6
	str_fusions = ['coexpression', 'cooccurence', 'database', 'experimental', 'fusion', 'neighborhood']
	adj_fusions = np.zeros((num_fusions, yeast_nums, yeast_nums))
	for idx in range(num_fusions):
		# path_fusion = '/media/userdisk1/jjpeng/xuehansheng/'+org+'_'+str_fusions[idx]+'_rwr.txt'
		path_fusion = 'data/rwr/'+org+'_'+str_fusions[idx]+'_rwr.txt'
		# adj_fusions[idx] = readRWR(path_fusion)
		# normalize
		adj_fusions[idx] = np.transpose(preprocessing.scale(readRWR(path_fusion)))
	return adj_fusions

def readGenes(filepath):
	dataSet = []
	for line in open(filepath):
		line = line.strip()
		dataSet.append(line)
	return dataSet

def readNetworks(filepath):
	network = np.zeros([yeast_nums, yeast_nums])
	for line in open(filepath):
		line = line.strip()
		temp = list(map(str,line.split('	')))
		network[int(temp[0])-1, int(temp[1])-1] = temp[2]
	return network

def readRWR(filepath):
	network = np.zeros([yeast_nums,yeast_nums])
	data = open(filepath, "r")
	network = [[float(v) for v in line.rstrip('\n').split('\t')] for line in data]

	return np.array(network)

def write_encoded_file(data, file_path):
	with open(file_path, "w") as f:
		for line in data:
			tempLine = ""
			for i in xrange(len(line)):
				tempLine = tempLine + str(line[i]) + "\t"
			tempLine = tempLine + "\n"
			f.write('%s'% tempLine)