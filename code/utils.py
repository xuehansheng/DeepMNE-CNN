# !/usr/bin/env python
# -*- coding: utf8 -*-
# author:xuehansheng(xhs1892@gmail.com)

import numpy as np
# import scipy.spatial.distance as ssd
from scipy.stats import pearsonr

yeast_nums = 6400

def extractConstraints(representation):
    num_ml = 0
    num_cl = 0
    
    mustlink_matrix = np.zeros((yeast_nums, yeast_nums))
    cannotlink_matrix = np.zeros((yeast_nums, yeast_nums))

    for i in xrange(len(representation)):
        for j in xrange(i):
            distance = pearsonr(representation[i],representation[j])[0] # pearson
            if distance <= 0.3:
                cannotlink_matrix[i,j] = 1.0
                cannotlink_matrix[j,i] = 1.0
                num_cl = num_cl + 1
            if distance >= 0.999:
                mustlink_matrix[i,j] = 1.0
                mustlink_matrix[j,i] = 1.0
                num_ml = num_ml + 1
    # print 'extract constraints:', num_ml, num_cl
    return mustlink_matrix, cannotlink_matrix

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


class SemiDataset:
        
    def __init__(self, data, constraints, constraints_):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._constraints = constraints
        self._constraints_ = constraints_
        self._num_examples = data.shape[0]
        pass

    @property
    def data(self):
        return self._data

    @property
    def constraints(self):
        return self._constraints

    @property
    def constraints_(self):
        return self._constraints_


    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)
            np.random.shuffle(idx)
            self._data = self.data[idx]
            self._constraints = self.constraints[idx]
            self._constraints_ = self.constraints_[idx]

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            start1 = start
            end1 = self._num_examples

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end =  self._index_in_epoch

            constraints_rest_part_1 = self.constraints[start1:end1, start1:end1]
            constraints_rest_part_2 = self.constraints[start1:end1, start:end]
            constraints_rest_part = np.hstack((constraints_rest_part_1, constraints_rest_part_2))

            constraints_rest_part_1_ = self.constraints_[start1:end1, start1:end1]
            constraints_rest_part_2_ = self.constraints_[start1:end1, start:end]
            constraints_rest_part_ = np.hstack((constraints_rest_part_1_, constraints_rest_part_2_))

            idx0 = np.arange(0, self._num_examples)
            np.random.shuffle(idx0)
            self._data = self.data[idx0]
            self._constraints = self.constraints[idx0]
            self._constraints_ = self.constraints_[idx0]

            data_new_part =  self._data[start:end]  
            constraints_new_part_1 = self._constraints[start:end, start:end]
            constraints_new_part_2 = self._constraints[start:end, start1:end1]
            constraints_new_part = np.hstack((constraints_new_part_1, constraints_new_part_2))

            constraints_new_part_1_ = self._constraints_[start:end, start:end]
            constraints_new_part_2_ = self._constraints_[start:end, start1:end1]
            constraints_new_part_ = np.hstack((constraints_new_part_1_, constraints_new_part_2_))

            data_batch = np.concatenate((data_rest_part, data_new_part), axis=0)
            constraints_batch = np.concatenate((constraints_rest_part, constraints_new_part), axis=0)
            constraints_batch_ = np.concatenate((constraints_rest_part_, constraints_new_part_), axis=0)
            return data_batch, constraints_batch, constraints_batch_
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._constraints[start:end, start:end], self._constraints_[start:end, start:end]
