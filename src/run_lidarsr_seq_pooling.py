#!/usr/bin/env python
from models.lidarsr_seq_models_pooling import *
from dataloader.lidarsr_seq import *

import tensorflow as tf

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



def test(iterate_count=50):
    weight_name = "/home/buaaren/lidar-sr/src/train_log/LidarSR/TCN_good/weights/weights.h5"
    test_data_input = load_test_seq_data()
    # load model
    model, _, _ = get_model('testing')
    model.load_weights(weight_name)

    this_test = np.empty([iterate_count, seq_length, image_rows_low, image_cols, channel_num], dtype=np.float32)
    test_data_prediction = np.empty([len(test_data_input), image_rows_high, image_cols, 2], dtype=np.float32)

    for i in range(len(test_data_input)):
    #for i in range(10):

        print('Processing {} th of {} images ... '.format(i, len(test_data_prediction)))

        for j in range(iterate_count):
            this_test[j] = test_data_input[i]

        this_prediction = model.predict(this_test, verbose=1)
        this_prediction_mean = np.mean(this_prediction, axis=0)
        this_prediction_var = np.std(this_prediction, axis=0)
        test_data_prediction[i,:,:,0:1] = this_prediction_mean
        test_data_prediction[i,:,:,1:2] = this_prediction_var


    np.save(os.path.join('TCN', test_set + '-' + model_name + '-from-' + str(image_rows_low) + '-to-' + str(image_rows_high) + '_prediction.npy'), test_data_prediction)



if __name__ == '__main__':

    # -> train network
    train()

    # test()
    
