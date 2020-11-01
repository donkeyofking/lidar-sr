#!/usr/bin/env python
import os
import time
#import _thread
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt



image_rows_low = 16
image_rows_high = 64
image_cols = 1024
channel_num = 1
sensor_noise = 0.03
max_range = 80.0
min_range = 2.0
normalize_ratio = 100.0
upscaling_factor = image_rows_high//image_rows_low

def get_low_res_from_high_res(high_res_data):
    low_res_index = range(0, image_rows_high, upscaling_factor)
    low_res_data = high_res_data[:,low_res_index]
    return low_res_data

training_data_file_name = '/media/buaaren/harddisk2t/ren/zeno/lidar_upsample_data/carla_ouster_range_image.npy'
# training_data_file_name = 'carla_ouster_range_image.npy'
# training_data_file_name = '/home/buaaren/lidar-sr-dataset/carla64.npy'
# training_data_file_name = '/home/buaaren/lidar-sr-dataset/ous64.npy'

i=0
def generate_data_from_file(batch_size=4):
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
    batch_no = full_res_data.shape[0]//batch_size
    while 1:
        global i
        y = full_res_data[i*batch_size:(i+1)*batch_size,:,:,:]
        x = get_low_res_from_high_res(y)
        # print("i:"+str(i))
        i = i+1
        if i>= batch_no:
            i=0
        # print(x.shape)
        yield (x,y)

if __name__=='__main__':
    generate_data_from_file()