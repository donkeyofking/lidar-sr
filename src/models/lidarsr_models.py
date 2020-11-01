#!/usr/bin/env python
import os
import datetime

import sys
sys.path.append("..")
from dataloader.lidarsr_data import *

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM, Input, Lambda, Conv2D, concatenate, Dense, Dropout, Activation, SeparableConv2D
from keras.layers import BatchNormalization, MaxPooling2D, UpSampling2D, Multiply, Add, Conv2DTranspose, AveragePooling2D
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
from keras import losses
from keras import metrics


os.environ['KERAS_BACKEND'] = 'tensorflow'
K.set_image_data_format('channels_last')
K.set_floatx('float32')

###########################################################################################
#################################   DataSet Parameters    #################################
###########################################################################################

upscaling_factor = 4



###########################################################################################
#################################   Model Specification   #################################
###########################################################################################
date_p = datetime.datetime.now().date()
time_p = datetime.datetime.now().time()
str_p = str(date_p) + str(time_p)
# Create Model 
model_name = 'UNet'
case_name = model_name + str_p
# home dir
project_name = 'LidarSR'
home_dir = os.path.expanduser('.')
root_dir = os.path.join(home_dir, 'train_log', project_name)
# automatically generate log and weight path
log_path = os.path.join(root_dir, case_name, 'logs')
weight_path = os.path.join(root_dir, case_name, 'weights',)
weight_name = os.path.join(weight_path, 'weights.h5')
history_path = os.path.join(root_dir,  case_name, 'history',)
history_name = case_name + '.txt'
history_name = os.path.join(history_path, history_name)

print('#'*30)
print('Using model:              {}'.format(model_name))
print('Trainig case:             {}'.format(case_name))
print('Log directory:            {}'.format(log_path))
print('Weight directory:         {}'.format(weight_path))
print('Weight name:              {}'.format(weight_name))
print('History directory:        {}'.format(history_path))
print('History name:             {}'.format(history_name))

path_lists = [log_path, weight_path, history_path]
for folder_name in path_lists:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

###########################################################################################
##############################        model UNet         ##################################
###########################################################################################

def UNet():

    def conv_block(input, filters=64, kernel_size=(3,3)):
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(input)
        x = BatchNormalization()(x)
        x = Activation(act_func)(x)
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(x)
        x = BatchNormalization()(x)
        x = Activation(act_func)(x)
        return x

    def up_block(input, filters=64, kernel_size=(3,3), strides=(1,1)):
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding='same', strides=strides, kernel_initializer=kernel_init)(input)
        x = BatchNormalization()(x)
        x = Activation(act_func)(x)
        return x

    filters = 64
    dropout_rate = 0.25
    act_func = 'relu'
    kernel_init = 'he_normal'

    inputs = Input((image_rows_low, image_cols, channel_num))

    # upscailing
    x0 = inputs
    for _ in range(int(np.log(upscaling_factor) / np.log(2))):
        x0 = up_block(x0, filters, strides=(2,1))

    x1 = conv_block(x0, filters)

    x2 = AveragePooling2D((2,2))(x1)
    x2 = Dropout(dropout_rate)(x2, training=True)
    x2 = conv_block(x2, filters*2)
     
    x3 = AveragePooling2D((2,2))(x2)
    x3 = Dropout(dropout_rate)(x3, training=True)
    x3 = conv_block(x3, filters*4)

    x4 = AveragePooling2D((2,2))(x3)
    x4 = Dropout(dropout_rate)(x4, training=True)
    x4 = conv_block(x4, filters*8)

    y4 = AveragePooling2D((2,2))(x4)
    y4 = Dropout(dropout_rate)(y4, training=True)
    y4 = conv_block(y4, filters*16)
    y4 = Dropout(dropout_rate)(y4, training=True)
    y4 = up_block(y4, filters*8, strides=(2,2))

    y3 = concatenate([x4, y4], axis=3)
    y3 = conv_block(y3, filters*8)
    y3 = Dropout(dropout_rate)(y3, training=True)
    y3 = up_block(y3, filters*4, strides=(2,2))

    y2 = concatenate([x3, y3], axis=3)
    y2 = conv_block(y2, filters*4)
    y2 = Dropout(dropout_rate)(y2, training=True)
    y2 = up_block(y2, filters*2, strides=(2,2))
 
    y1 = concatenate([x2, y2], axis=3)
    y1 = conv_block(y1, filters*2)
    y1 = Dropout(dropout_rate)(y1, training=True)
    y1 = up_block(y1, filters, strides=(2,2))

    y0 = concatenate([x1, y1], axis=3)
    y0 = conv_block(y0, filters)

    outputs = Conv2D(1, (1, 1), activation=act_func)(y0)

    def PSNR(y_true, y_pred):
        max_pixel= 1.0
        return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))) / 2.303

    from keras_contrib.losses import DSSIMObjective
    loss_func = DSSIMObjective()

    model = Model(inputs=inputs, outputs=outputs)
    # model.compile(optimizer=Adam(lr=0.0001, decay=0.00001),loss=loss_func ,metrics =['accuracy', 'mse' , 'mae', PSNR, loss_func ] )
    model.compile(optimizer=Adam(lr=0.0001, decay=0.00001),loss='mse' ,metrics =['accuracy', 'mse' , 'mae', PSNR, loss_func ] )

    model.summary()

    return model


###########################################################################################
#################################   some functions    #####################################
###########################################################################################
def create_case_dir(type_name):
    # tensorboard
    model_checkpoint = None
    tensorboard = None
    os.system('killall tensorboard')
    # create tensorboard checkpoint
    if type_name == 'training':
        model_checkpoint = ModelCheckpoint(weight_name, save_best_only=True, period=1)
        tensorboard = TensorBoard(log_dir=log_path)
        # run tensorboard
        command = 'tensorboard --logdir=' + os.path.join(root_dir, 'logs') + ' &'
        os.system(command)
        # delete old log files
        for the_file in os.listdir(log_path):
            file_path = os.path.join(log_path, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    return model_checkpoint, tensorboard

def get_model(type_name='training'):
    # create case dir
    model_checkpoint, tensorboard = create_case_dir(type_name)
    # create default model
    model = None
    # Choose Model
    if model_name == 'UNet':
        model = UNet()
    return model, model_checkpoint, tensorboard