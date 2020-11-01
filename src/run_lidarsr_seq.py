#!/usr/bin/env python
from models.lidarsr_seq_models import *
from dataloader.lidarsr_seq import *

import tensorflow as tf
from time import *
import pickle

def train():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    print('Compiling model...     ')
    model, model_checkpoint, tensorboard = get_model('training')

    print('Training model...      ')
    history = model.fit_generator(generate_seq_data_from_file(),
                                steps_per_epoch=400, #400 
                                epochs=20, 
                                max_queue_size=10,
                                verbose=1,
                                shuffle=True,
                                workers=1,
                                validation_data=generate_seq_data_from_file(),
                                validation_steps=100,
                                callbacks=[model_checkpoint, tensorboard])

    model.save(weight_name)

    with open(history_name, 'wb') as file_txt:
        pickle.dump(history.history, file_txt)


import numpy as np
import random

def test_time(iterate_count=50):
    weight_name = "/home/buaaren/lidar-sr/src/train_log/LidarSR/TCN_good/weights/weights.h5"
    # load model
    model, _, _ = get_model('testing')
    model.load_weights(weight_name)
    test_data = np.load(training_data_file_name)
    print(test_data.shape)
    # this_test = np.empty([iterate_count, seq_length, image_rows_low, image_cols, channel_num], dtype=np.float32)
    # test_data_prediction = np.empty([len(test_data_input), image_rows_high, image_cols, 2], dtype=np.float32)
    begin_time = time()
    for i in range(1000):
    #for i in range(10):
        y = []
        j = random.randint(0,test_data.shape[0]-16)
        index_start = j
        index_end   = j + 16
        y.append(test_data[index_start:index_end ,:,:,:])
        y = np.array(y)
        y = get_low_res_from_high_res(y)
        print(y.shape)
        print('Processing {} th of {} images ... '.format(i, 1000))

        # for j in range(iterate_count):
        #     this_test[j] = test_data_input[i]
        every_time = time()
        this_prediction = model.predict(y, verbose=1)
        pause_time = time()
        print(pause_time-every_time)
        # this_prediction_mean = np.mean(this_prediction, axis=0)
        # this_prediction_var = np.std(this_prediction, axis=0)
        # test_data_prediction[i,:,:,0:1] = this_prediction_mean
        # test_data_prediction[i,:,:,1:2] = this_prediction_var
    end_time = time()
    run_time = end_time - begin_time
    print('run time')
    print(run_time)

    # np.save(os.path.join('TCN', test_set + '-' + model_name + '-from-' + str(image_rows_low) + '-to-' + str(image_rows_high) + '_prediction.npy'), test_data_prediction)


def test_one(iterate_count=50):
    weight_name = "/home/buaaren/lidar-sr/src/train_log/LidarSR/TCN_good/weights/weights.h5"
    # load model
    model, _, _ = get_model('testing')
    model.load_weights(weight_name)
    test_data = np.load(training_data_file_name)
    print(test_data.shape)
    batch_no = test_data.shape[0]-16
    # this_test = np.empty([iterate_count, seq_length, image_rows_low, image_cols, channel_num], dtype=np.float32)
    # test_data_prediction = np.empty([len(test_data_input), image_rows_high, image_cols, 2], dtype=np.float32)
    result = []
    for i in range(batch_no):
    #for i in range(10):
        y = []
        index_start = i
        index_end   = i + 16
        y.append(test_data[index_start:index_end ,:,:,:])
        y = np.array(y)
        y = get_low_res_from_high_res(y)
        print(y.shape)
        print('Processing {} th of {} images ... '.format(i, batch_no))

        # for j in range(iterate_count):
        #     this_test[j] = test_data_input[i]
        this_prediction = model.predict(y, verbose=1)
        print(this_prediction.shape)
        result.append(this_prediction)
        # this_prediction_mean = np.mean(this_prediction, axis=0)
        # this_prediction_var = np.std(this_prediction, axis=0)
        # test_data_prediction[i,:,:,0:1] = this_prediction_mean
        # test_data_prediction[i,:,:,1:2] = this_prediction_var
    result = np.array(result)
    np.save('prediction64.npy', result)

if __name__ == '__main__':

    # -> train network
    # train()

    test_one()
    
