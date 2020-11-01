#!/usr/bin/env python
import os
import time
#import _thread
import math
import cv2
import numpy as np
import random

image_rows_low = 16
image_rows_high = 64
image_cols = 1024
channel_num = 1
sensor_noise = 0.03
max_range = 80.0
min_range = 2.0
normalize_ratio = 100.0 # all point in range(2,80)
upscaling_factor = image_rows_high//image_rows_low
batch_size_ = 1
seq_length = 16


def get_low_res_from_high_res(high_res_data):
    low_res_index = range(0, image_rows_high, upscaling_factor)
    low_res_data = high_res_data[:,:,low_res_index,:,:]
    return low_res_data

# training_data_file_name = '/media/buaaren/harddisk2t/ren/zeno/lidar_upsample_data/carla_ouster_range_image.npy'
# testing_data_file_name = '/media/buaaren/harddisk2t/ren/zeno/lidar_upsample_data/ouster_range_image.npy'
training_data_file_name = '/home/buaaren/lidar-sr-dataset/carla64.npy'


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
        # print(index_start, index_end)
        y = []
        gt = []
        for b in range(batch_size):
            y.append(full_res_data[index_start + b:index_start + b +  seq_size ,:,:,:])
            gt.append(full_res_data[index_start + b + seq_size-1])
        y = np.array(y)
        x = get_low_res_from_high_res(y)
        gt = np.array(gt)
        # print(x.shape)
        # print(gt.shape)
        i += 1
        if i >= batch_no:
            i=0
        yield (x,gt)



def generate_test_seq_data_from_file(batch_size=batch_size_, seq_size=seq_length):
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
        # print(index_start, index_end)
        y = []
        gt = []
        for b in range(batch_size):
            y.append(full_res_data[index_start + b:index_start + b +  seq_size ,:,:,:])
            gt.append(full_res_data[index_start + b + seq_size-1])
        y = np.array(y)
        x = get_low_res_from_high_res(y)
        gt = np.array(gt)
        # print(x.shape)
        # print(gt.shape)
        i += 1
        if i >= batch_no:
            i=0
        yield (x,gt)

if __name__=='__main__':
    # generate_seq_data_from_file()
    # load_test_seq_data()
    generate_test_seq_data_from_file()