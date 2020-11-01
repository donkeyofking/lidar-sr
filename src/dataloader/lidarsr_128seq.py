#!/usr/bin/env python
import os
import time
#import _thread
import math
import cv2
import numpy as np
import random

image_rows_low = 32
image_rows_high = 128
image_cols = 1808 #ruby128 is 1800, for //
channel_num = 1
sensor_noise = 0.03
max_range = 160.0
min_range = 3.0
normalize_ratio = 160.0 # all point in range(3,160)
upscaling_factor = image_rows_high//image_rows_low
batch_size_ = 1
seq_length = 16


def get_low_res_from_high_res(high_res_data):
    low_res_index = range(0, image_rows_high, upscaling_factor)
    low_res_data = high_res_data[:,:,low_res_index,:,:]
    return low_res_data

training_data_file_name = '/home/buaaren/lidar-sr-dataset/carla128.npy'
# testing_data_file_name = '/home/buaaren/lidar-sr-dataset/ruby128.npy'


def generate_seq_data_from_file(batch_size=batch_size_, seq_size=seq_length):
    full_res_data = np.load(training_data_file_name)
    print(full_res_data.shape)
    print('----------------------------')
    full_res_data = full_res_data.astype(np.float32, copy=True)
    noise = np.random.normal(0, sensor_noise, full_res_data.shape)
    noise[full_res_data == 0] = 0
    full_res_data = full_res_data + noise
    # apply sensor range limit
    full_res_data[full_res_data > max_range] = 0
    full_res_data[full_res_data < min_range] = 0
    # normalize data
    full_res_data = full_res_data / normalize_ratio
    batch_no = (full_res_data.shape[0] - seq_size + 1) // batch_size
    print(batch_no)
    global i
    i=0
    seq_file_no = seq_size + batch_size -1
    while 1:
        # random sample from file , after commenting it sequence sample from file
        i = random.randint(0,batch_no-1)
        # print(i)
        index_start =  i    *  batch_size 
        index_end   =  i    *  batch_size + seq_file_no -1
        y = []
        gt = []
        for b in range(batch_size):
            y.append(full_res_data[index_start + b:index_start + b +  seq_size ,:,:,:])
            gt.append(full_res_data[index_start + b + seq_size-1])
        y = np.array(y)
        x = get_low_res_from_high_res(y)
        gt = np.array(gt)
        # print(index_start, index_end)
        # print(x.shape)
        # print(gt.shape)
        i += 1
        if i >= batch_no:
            i=0
        yield (x,gt)



def load_test_seq_data():
    full_res_data = np.load(testing_data_file_name)
    print(full_res_data.shape)
    print('---------testing data-------------------')
    full_res_data = full_res_data.astype(np.float32, copy=True)
    noise = np.random.normal(0, sensor_noise, full_res_data.shape)
    noise[full_res_data == 0] = 0
    full_res_data = full_res_data + noise
    # apply sensor range limit
    full_res_data[full_res_data > max_range] = 0
    full_res_data[full_res_data < min_range] = 0
    # normalize data
    full_res_data = full_res_data / normalize_ratio
    low_res_index = range(0, image_rows_high, upscaling_factor)
    test_data_input = full_res_data[:,low_res_index,:,:]
    dataX = []
    for i in range(len(test_data_input)-seq_length):
        X = test_data_input[i:i + seq_length]
        dataX.append(X)
    dataX = np.array(dataX)
    print(dataX.shape)
    return dataX

if __name__=='__main__':
    generate_seq_data_from_file()
    # load_test_seq_data()