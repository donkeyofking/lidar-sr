#!/usr/bin/env bash
nvcc=/usr/local/cuda-10.0/bin/nvcc
cudalib=/usr/local/cuda-10.0/lib64/
#TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_INC=/home/buaaren/anaconda3/envs/lidar-sr/lib/python3.7/site-packages/tensorflow/include
TF_LIB=/home/buaaren/anaconda3/envs/lidar-sr/lib/python3.7/site-packages/tensorflow

$nvcc tf_nndistance_g.cu -c -o tf_nndistance_g.cu.o -std=c++11  -I $TF_INC -DGOOGLE_CUDA=1\
 -x cu -Xcompiler -fPIC -O2

g++ tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -std=c++11 -shared -fPIC -I $TF_INC \
-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -lcudart -L $cudalib -O2
