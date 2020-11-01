#!/usr/bin/env python

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from models.lidarsr_pano_models import *
from dataloader.lidarsr_pano_data import *

import tensorflow as tf

import pickle
from keras import backend as K


def train():


    print('Compiling model...     ')
    model, model_checkpoint, tensorboard = get_model('training')
    

    print('Training model...      ')
    history = model.fit_generator(generate_data_from_path(4),
                                steps_per_epoch=400, 
                                epochs=10, 
                                max_queue_size=10,
                                verbose=1,
                                # shuffle=True,
                                workers=1,
                                validation_data=generate_data_from_path(4),
                                validation_steps=100,
                                callbacks=[model_checkpoint, tensorboard])
    print("training finished")
    model.save(weight_name)
    print("save history file")
    with open(history_name, 'wb') as file_txt:
        pickle.dump(history.history, file_txt)



def test():
    if not os.path.exists(pred64):
        os.makedirs(pred64)
    model, _, _ = get_model('testing')
    # weight_name = "/home/buaaren/lidar-sr/src/train_log/LidarSR_panorama/latest/weights/weights.h5"
    model.load_weights(weight_name)
    for testfile in lidar16_test_files:
        x = []
        testdata = point_cloud_to_panorama_2(np.load(os.path.join(test16,testfile)),
                                            v_res = v_res,
                                            h_res = h_res,
                                            v_height = v_height,
                                            d_range = (0,80),
                                            )
        testdata = testdata[np.newaxis, : ,:, :]
        # testdata = testdata[np.newaxis, : ,:, np.newaxis]

        print(testdata.shape)
        this_prediction = model.predict(testdata, verbose=1)
        print(this_prediction.shape)
        this_prediction = np.squeeze(this_prediction)
        print(this_prediction.shape)
        np.save(os.path.join(pred64,testfile),this_prediction)
    print("Test finished.")



if __name__ == '__main__':

    # -> train network
    train()
    test()


