# !/usr/bin/env python
# -*- coding: utf8 -*-
# author:xuehansheng(xhs1892@gmail.com)

import numpy as np

def readGenes(filepath):
	dataSet = []
	for line in open(filepath):
		line = line.strip()
		dataSet.append(line)
	return dataSet

def readAdjacency(filepath, len_term, len_gene):
	mips_anno = np.zeros([len_term, len_gene])
	for line in open(filepath):
		line = line.strip()
		# temp = map(str,line.split('	'))
		temp = list(map(str,line.split('	')))
		mips_anno[int(temp[1])-1, int(temp[0])-1] = 1.0
	return mips_anno

def ismember(genes, mips_genes):
	result = np.zeros(len(genes))
	for i in range(len(genes)):
		if genes[i] in mips_genes:
			result[i] = 1.0
	return result

def get_genesfilt(genes, filt_):
	genes_ = []
	for i in range(len(genes)):
		if filt_[i] == 1.0:
			genes_.append(genes[i])
	return genes_

def get_s2goind(genes, genes_filt_):
	s2goind = []
	for i in range(len(genes_filt_)):
		index = genes.index(genes_filt_[i])
		s2goind.append(index)
	return s2goind

def get_anno(mips_anno, s2goind, filt_):
	anno = np.zeros([len(mips_anno), len(filt_)])
	filt_index = [i for i,x in enumerate(filt_) if x == 1.0]
	for i in range(len(s2goind)):
		for j in range(len(mips_anno)):
			anno[j,filt_index[i]] = mips_anno[j, s2goind[i]]
	return anno

def load_mips(level_):
	yeast_genes_file = './data/networks/yeast/yeast_string_genes.txt'
	yeast_genes_level_file = './data/annotations/yeast/yeast_mips_' + str(level_) + '_genes.txt'
	yeast_terms_level_file = './data/annotations/yeast/yeast_mips_' + str(level_) + '_terms.txt'
	yeast_level_adjacency_file = './data/annotations/yeast/yeast_mips_' + str(level_) + '_adjacency.txt'

	yeast_genes = readGenes(yeast_genes_file)
	yeast_mips_genes = readGenes(yeast_genes_level_file)
	filt = ismember(yeast_genes, yeast_mips_genes)

	yeast_mips_terms = readGenes(yeast_terms_level_file)
	mips_anno = readAdjacency(yeast_level_adjacency_file, len(yeast_mips_terms), len(yeast_mips_genes))

	genes_filt = get_genesfilt(yeast_genes, filt)

	s2goind = get_s2goind(yeast_mips_genes, genes_filt)
	anno = get_anno(mips_anno, s2goind, filt)

	return anno

# if __name__ == '__main__':
# 	anno = load_mips()

# 	print anno[:,10:20]




