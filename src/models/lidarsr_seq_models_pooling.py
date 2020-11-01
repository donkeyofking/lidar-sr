#!/usr/bin/env python
import os
import datetime
import inspect

import sys
sys.path.append("..")
from dataloader.lidarsr_seq import *

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM, Input, Lambda, Conv2D, concatenate, Dense, Dropout, Activation, SeparableConv2D, AtrousConvolution2D
from keras.layers import BatchNormalization, MaxPooling2D, UpSampling2D, Multiply, Add, Conv2DTranspose, AveragePooling2D
from keras.layers import Conv3DTranspose
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
from keras import losses
from keras import metrics

from tcn import TCN, tcn_full_summary

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
# model_name = 'UNet'
model_name = 'TCN'
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
##############################        model TCN          ##################################
###########################################################################################

def TCN_net():


    def conv_block(input, filters=32, kernel_size=(3,3)):
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal')(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def BandA(input):
        x = BatchNormalization()(input)
        x = Activation('relu')(x)
        return x

    inputs = Input((seq_length, image_rows_low, image_cols, channel_num))

    # upscailing
    x0 = inputs
        # split in seq_length
    split_x = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': seq_length})(inputs)
        #　shared parameteres layer
    up0 = Conv2DTranspose(filters=32, kernel_size=(3,3), padding='same', strides=(2,1), kernel_initializer='he_normal')
    up1 = Conv2DTranspose(filters=32, kernel_size=(3,3), padding='same', strides=(2,1), kernel_initializer='he_normal')

    output1=[]# 0,1,2,3,4,5,6...15
    for i in range(seq_length):
        slice = split_x[i]
        slice = Lambda(tf.squeeze, arguments={'axis': 1,})(slice)
        slice = up0(slice)
        slice = BatchNormalization()(slice)
        slice = Activation('relu')(slice)
        slice = up1(slice)
        slice = BatchNormalization()(slice)
        slice = Activation('relu')(slice)
        output1.append(slice)

    output2=[]# 0,1,2,3,4,5,6,7
    conv1 = Conv2D(filters=64, kernel_size=(3,3),  padding='same', kernel_initializer='he_normal')
    conv2 = Conv2D(filters=64, kernel_size=(3,3),  padding='same', kernel_initializer='he_normal')

    for i in range(int(seq_length/2)):
        slice = concatenate([output1[2*i],output1[2*i+1]],axis=3)
        slice = conv1(slice)
        slice = BatchNormalization()(slice)
        slice = Activation('relu')(slice)
        slice = conv2(slice)
        slice = BatchNormalization()(slice)
        slice = Activation('relu')(slice)
        slice = MaxPooling2D(pool_size=(2,2))(slice)
        output2.append(slice)

    output3=[] # 0,1,2,3
    conv3 = Conv2D(filters=128, kernel_size=(3,3),  padding='same', kernel_initializer='he_normal')
    conv4 = Conv2D(filters=128, kernel_size=(3,3),  padding='same', kernel_initializer='he_normal')
    for i in range(int(seq_length/4)):
        slice = concatenate([output2[i*2],output2[2*i+1]],axis=3)
        slice = conv3(slice)
        slice = BatchNormalization()(slice)
        slice = Activation('relu')(slice)
        slice = conv4(slice)
        slice = BatchNormalization()(slice)
        slice = Activation('relu')(slice)
        slice = MaxPooling2D(pool_size=(2,2))(slice)
        output3.append(slice)

    output4=[] # 0,1
    conv5 = Conv2D(filters=256, kernel_size=(3,3),  padding='same', kernel_initializer='he_normal')
    conv6 = Conv2D(filters=256, kernel_size=(3,3),  padding='same', kernel_initializer='he_normal')
    for i in range(int(seq_length/8)):
        slice = concatenate([output3[i*2],output3[2*i+1]],axis=3)
        slice = conv5(slice)
        slice = BatchNormalization()(slice)
        slice = Activation('relu')(slice)
        slice = conv6(slice)
        slice = BatchNormalization()(slice)
        slice = Activation('relu')(slice)
        slice = MaxPooling2D(pool_size=(2,2))(slice)
        output4.append(slice)

    output5=[] # 0
    conv7 = Conv2D(filters=512, kernel_size=(3,3),  padding='same', kernel_initializer='he_normal')
    conv8 = Conv2D(filters=512, kernel_size=(3,3),  padding='same', kernel_initializer='he_normal')
    for i in range(int(seq_length/16)):
        slice = concatenate([output4[i*2],output4[2*i+1]],axis=3)
        slice = conv7(slice)
        slice = BatchNormalization()(slice)
        slice = Activation('relu')(slice)
        slice = conv8(slice)
        slice = BatchNormalization()(slice)
        slice = Activation('relu')(slice)
        slice = MaxPooling2D(pool_size=(2,2))(slice)
        output5.append(slice)

    highest_map = output5[0]
    highest_map = Conv2D(filters=1024, kernel_size=(3,3), padding='same', kernel_initializer='he_normal')(highest_map)
    highest_map = BatchNormalization()(highest_map)
    highest_map = Activation('relu')(highest_map)
    highest_map = Conv2D(filters=1024, kernel_size=(3,3), padding='same', kernel_initializer='he_normal')(highest_map)
    highest_map = BatchNormalization()(highest_map)
    highest_map = Activation('relu')(highest_map)

    up1_map = Conv2DTranspose(filters=512, kernel_size=(3,3), padding='same', strides=(2,2), kernel_initializer='he_normal')(highest_map)
    up1_map = BandA(up1_map)
    up1_concat = concatenate([up1_map,output4[1]],axis = 3)
    up1_concat = Conv2D(filters=512, kernel_size=(3,3), padding='same', kernel_initializer='he_normal')(up1_concat)
    up1_concat = BandA(up1_concat)
    up1_concat = Conv2D(filters=512, kernel_size=(3,3), padding='same', kernel_initializer='he_normal')(up1_concat)
    up1_concat = BandA(up1_concat)

    up2_map = Conv2DTranspose(filters=256, kernel_size=(3,3), padding='same', strides=(2,2), kernel_initializer='he_normal')(up1_concat)
    up2_map = BandA(up2_map)
    up2_concat = concatenate([up2_map,output3[3]],axis = 3)
    up2_concat = Conv2D(filters=256, kernel_size=(3,3), padding='same', kernel_initializer='he_normal')(up2_concat)
    up2_concat = BandA(up2_concat)
    up2_concat = Conv2D(filters=256, kernel_size=(3,3), padding='same', kernel_initializer='he_normal')(up2_concat)
    up2_concat = BandA(up2_concat)

    up3_map = Conv2DTranspose(filters=128, kernel_size=(3,3), padding='same', strides=(2,2), kernel_initializer='he_normal')(up2_concat)
    up3_map = BandA(up3_map)
    up3_concat = concatenate([up3_map,output2[7]],axis = 3)
    up3_concat = Conv2D(filters=128, kernel_size=(3,3), padding='same', kernel_initializer='he_normal')(up3_concat)
    up3_concat = BandA(up3_concat)
    up3_concat = Conv2D(filters=128, kernel_size=(3,3), padding='same', kernel_initializer='he_normal')(up3_concat)
    up3_concat = BandA(up3_concat)

    up4_map = Conv2DTranspose(filters= 64, kernel_size=(3,3), padding='same', strides=(2,2), kernel_initializer='he_normal')(up3_concat)
    up4_map = BandA(up4_map)
    up4_concat = concatenate([up4_map,output1[15]],axis = 3)
    up4_concat = Conv2D(filters=64, kernel_size=(3,3), padding='same', kernel_initializer='he_normal')(up4_concat)
    up4_concat = BandA(up4_concat)
    up4_concat = Conv2D(filters=64, kernel_size=(3,3), padding='same', kernel_initializer='he_normal')(up4_concat)
    up4_concat = BandA(up4_concat)

    lowest_map = Conv2DTranspose(filters=64, kernel_size=(3,3), padding='same', strides=(2,2), kernel_initializer='he_normal')(up3_concat)

    print(K.int_shape(lowest_map))
    gen_image = Conv2D(1, (1, 1), activation='relu')(lowest_map)
    print(K.int_shape(gen_image))

    outputs = gen_image

    def PSNR(y_true, y_pred):
        max_pixel= 1.0
        return -(10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))) / 2.303

    from keras_contrib.losses import DSSIMObjective
    loss_func = DSSIMObjective()

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr=0.0001, decay=0.00001),loss=loss_func ,metrics =['accuracy', 'mse' , 'mae', PSNR, loss_func ] )
    # model.compile(optimizer=Adam(lr=0.0001, decay=0.00001),loss=PSNR ,metrics =['accuracy', 'mse' , 'mae', PSNR, loss_func ] )

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
    if model_name == 'TCN':
        model = TCN_net()
    return model, model_checkpoint, tensorboard