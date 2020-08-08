#!/usr/bin/env python
from models.lidarsr_models import *
from dataloader.lidarsr_data import *

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
    history = model.fit_generator(generate_data_from_file(),
                                steps_per_epoch=200, 
                                epochs=20, 
                                max_queue_size=10,
                                verbose=1,
                                shuffle=True,
                                workers=1,
                                validation_data=generate_data_from_file(),
                                validation_steps=100,
                                callbacks=[model_checkpoint, tensorboard])

    model.save(weight_name)

    with open(history_name, 'wb') as file_txt:
        pickle.dump(history.history, file_txt)


if __name__ == '__main__':

    # -> train network
    train()


