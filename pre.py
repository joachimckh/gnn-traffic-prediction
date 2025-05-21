#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:11:08 2024

@author: dliu
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


dataname = 'All2020'
data = np.load(f'{dataname}.npz')['data']
data = data.transpose(1,0,2,3)

# dataname = 'may2020_tmp'
# data = np.load(f'{dataname}.npz')['data']

save = True
len_seq = 12
days = 3
all_samples = []

day_list = []
for i in range(data.shape[1]-days+1):
    day_list.append([i+idx for idx in range(days)])
    
# for idx_day in combinations(range(10),days):
# for idx_day in [(0,1,2),(1,2,3),(2,3,4),(3,4,5),(4,5,6),(5,6,7),(6,7,8),(7,8,9)]:
for idx_day in day_list:
    for idx_t in range(data.shape[3]-len_seq*2):
        all_samples.append(data[:,idx_day,0,idx_t:idx_t+len_seq*2])
all_samples = np.array(all_samples)


# ### time embedding ###
# time_label = []
# for idx_day in day_list:
#     for idx_t in range(data.shape[3]-len_seq*2):
#         time_label.append(data[:,idx_day,1,idx_t:idx_t+len_seq*2])
# time_label = np.array(time_label)
# all_samples = np.concatenate((time_label[:,:,[0],:]/time_label.max(),all_samples),axis=2)
# ######################


### time embedding ###
time_label = []
time_embedding = np.zeros_like(data)
time_embedding[:,:,1,:] = np.linspace(0,1,data.shape[-1])
for idx_day in day_list:
    for idx_t in range(time_embedding.shape[3]-len_seq*2):
        time_label.append(time_embedding[:,idx_day,1,idx_t:idx_t+len_seq*2])
time_label = np.array(time_label)
all_samples = np.concatenate((time_label[:,:,[0],:]/time_label.max()*0.5,all_samples),axis=2)
######################


split_line1 = int(len(all_samples) * 0.6)
split_line2 = int(len(all_samples) * 0.8)


training_set = all_samples[:split_line1]
validation_set = all_samples[split_line1: split_line2]
testing_set = all_samples[split_line2:]


train_x = training_set[:,:,:,:len_seq]
val_x = validation_set[:,:,:,:len_seq]
test_x = testing_set[:,:,:,:len_seq]

train_target = training_set[:,:,-1,len_seq:]
val_target = validation_set[:,:,-1,len_seq:]
test_target = testing_set[:,:,-1,len_seq:]



def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same
    mean = train.mean(axis=(0,1,3), keepdims=True)
    std = train.std(axis=(0,1,3), keepdims=True)
    print('mean.shape:',mean.shape)
    print('std.shape:',std.shape)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm

(stats, train_x_norm, val_x_norm, test_x_norm) = normalization(train_x, val_x, test_x)


all_data = {
    'train': {
        'x': train_x_norm,
        'target': train_target,
        'timestamp': 1,
    },
    'val': {
        'x': val_x_norm,
        'target': val_target,
        'timestamp': 1,
    },
    'test': {
        'x': test_x_norm,
        'target': test_target,
        'timestamp': 1,
    },
    'stats': {
        '_mean': stats['_mean'],
        '_std': stats['_std'],
    }
}
print("Data shape")
print("-"*10)
print("train_x",all_data["train"]["x"].shape)
print("train_tar",all_data["train"]["target"].shape)
print("-"*10)
print("val_x",all_data["val"]["x"].shape)
print("val_tar",all_data["val"]["target"].shape)
print("-"*10)
print("test_x",all_data["test"]["x"].shape)
print("test_tar",all_data["test"]["target"].shape)
print("-"*10)
if save:
    filename = os.path.join(f'./{dataname}' + '_astcgn')
    print('save file:', filename)
    np.savez_compressed(filename,
                        train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                        train_timestamp=all_data['train']['timestamp'],
                        val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                        val_timestamp=all_data['val']['timestamp'],
                        test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                        test_timestamp=all_data['test']['timestamp'],
                        mean=all_data['stats']['_mean'], std=all_data['stats']['_std']
                        )

