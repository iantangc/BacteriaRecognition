from __future__ import print_function, division
import keras
from keras.layers import Dense, Add, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.layers import MaxPooling2D, AveragePooling2D, ZeroPadding2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras.datasets import cifar10
import numpy as np
import os
import sys
import h5py

# import pydot
# import graphviz
# from IPython.display import SVG

def composite_conv_layer(inputs,
                num_filters=16,
                kernel_size=3,
                strides=1,
                activation='relu',
                batch_normalization=True,
                conv_first=True,
                regularization_factor=1e-4,
                layer_name=''):
    """
    Builds a layer for a typical ResNet
    
    Structure:
    Convolution 2D --> Batch Norm --> Activation layer or
    Batch Norm --> Activation layer --> Convolution 2D
    
    # Arguments
        (tensor) inputs      : input tensor from input image or previous layer
        (int)    num_filters : Conv2D number of filters
        (int)    kernel_size : Conv2D square kernel dimensions
        (int)    strides     : Conv2D square stride dimensions
        (string) activation  : activation function name as defined in Keras
        (bool)   batch_normalization : whether to include batch normalization
        (bool)   conv_first  : (True) conv --> BN --> activation  or
                               (False) BN --> activation --> conv 
        (float)  regularization_factor: Regularization factor of L2 regularization
        (string) layer_name  : User-specified name for the layer

    # Returns
        (tensor) x           : tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size = kernel_size,
                  strides = strides,
                  padding = 'same',
                  kernel_initializer = 'he_normal',
                  kernel_regularizer = l2(regularization_factor),
                  name = layer_name + '_CONV')

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization(name = layer_name + '_BN')(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def res_block(inputs, kernel_size = 3, strides = 1, force_shortcut_convolutional = False, num_filters_list = [128, 128, 256], block_name = ''):
    """
    Builds a ResNet block with skip connection (bottleneck structure)
    
    Structure:
    Identity Block: 
      /-> (CONV2D > BN > ReLU) --> (CONV2D > BN > ReLU) --> (CONV2D > BN) -\
     X                                                                     (+) --> ReLU
      \--------------------------------------------------------------------/
    
    Convolution / Downsampling Block:
    
      /-> (CONV2D* > BN > ReLU) --> (CONV2D > BN > ReLU) --> (CONV2D > BN) -\
     X                                                                      (+) --> ReLU
      \-----------------------------(CONV2D* > BN)--------------------------/
   
    (Note: both CONV2D* shares the same stride)
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    
    Convolution 2D --> Batch Norm --> Activation layer or
    Batch Norm --> Activation layer --> Convolution 2D
    
    # Arguments
        (tensor) inputs      : input tensor from previous layer
        (int)    kernel_size : Conv2D square kernel dimensions of the middle convolution layer in the main path
        (int)    strides     : Conv2D square stride dimensions of the first convolution layer in the main path 
                               and the skip connection, forcing convolution in skip connection if strides != 1
        (bool)   force_shortcut_convolutional: Specify whether the skip connection is convoluted when strides = 1
        (list[int]) num_filters_list: list of size 3, 
                                      the number of filters used in each convolution layer, starting from the beginning
                                      the output channel of this block is the same as the third integer
        (string) block_name  : User-specified name for the block
        
    # Returns
        (tensor) X           : tensor as input to the next layer / block
    """
    
    # defining name basis
    block_name_base = 'RES_' + block_name

    # Retrieve numbers of filters
    num_filters_1, num_filters_2, num_filters_3 = num_filters_list
    
    # Save the input value for the shortcut
    X = inputs
    X_shortcut = inputs

    ##### MAIN PATH #####
    
    # First component of main path, 1 x 1 x f1, strides = s
    X = composite_conv_layer(X, num_filters = num_filters_1, kernel_size = 1, strides = strides,
            activation='relu', batch_normalization=True, conv_first=True, layer_name = block_name_base + '_main_1')

    # Second component of main path, k x k x f2, strides = 1
    X = composite_conv_layer(X, num_filters = num_filters_2, kernel_size = kernel_size, strides = 1,
            activation='relu', batch_normalization=True, conv_first=True, layer_name = block_name_base + '_main_2')

    # Third component of main path, 1 x 1 x f3, strides = 1, No activation
    X = composite_conv_layer(X, num_filters = num_filters_3, kernel_size = 1, strides = 1,
            activation = None, batch_normalization=True, conv_first=True, layer_name = block_name_base + '_main_3')

    ##### SHORTCUT PATH #####
    if strides != 1 or force_shortcut_convolutional:
        # Shortcut, 1 x 1 x f3, strides = s
        X_shortcut = composite_conv_layer(X_shortcut, num_filters = num_filters_3, kernel_size = 1, strides = strides,
                        activation = None, batch_normalization=True, conv_first=True, layer_name = block_name_base + '_shortcut')
    
    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
    
    return X
    
def resnet_model(input_shape, num_output_classes, parameters = []):
    
    """
    Builds a ResNet Model (v2)
    
    First block is a standard convolutional layer followed by max pooling
    
    Resnet blocks are then stacked as stages according to the specification
    
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides = 2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    
    # Arguments
        (tensor)   input_shape       : shape of input image tensor (3 dimensional)
        (int)      num_resnet_stages : number of resnet stages
        (int)      num_output_classes       : number of classes (CIFAR10 has 10)
    # Returns
        (Model)    model : Keras model instance using ResNet architecture
    """
    
    X_input = Input(shape=input_shape)
    X = ZeroPadding2D(padding=(0, 0))(X_input)
    
    layer_count = 0
    
    for si in range(len(parameters)):
        stage_name_base_str = 'stage' + str(si) + '_'
        for param_dict in parameters[si]:
            unit_type = param_dict['unit_type']
            if unit_type == 'std_conv':
                layer_count += 1
                X = composite_conv_layer(X, num_filters = param_dict['f'], kernel_size = param_dict['k'], strides = param_dict['s'],
                        activation = 'relu', batch_normalization=True, conv_first=True, layer_name = stage_name_base_str + param_dict['name'])
            elif unit_type == 'max_pool':
                X = MaxPooling2D(pool_size = param_dict['k'], strides = param_dict['s'], padding='valid')(X)
            elif unit_type == 'res_block':
                layer_count += 3
                X = res_block(X, kernel_size = param_dict['k'], strides = param_dict['s'], force_shortcut_convolutional = param_dict['force_conv'], 
                              num_filters_list = param_dict['fs'], block_name = stage_name_base_str + param_dict['name'])
            elif unit_type == 'bn':
                X = BatchNormalization(name = stage_name_base_str + '_BN')(X)
            elif unit_type == 'activation':
                X = Activation(param_dict['activation'])(X)
            elif unit_type == 'global_avg_pool':
                X = GlobalAveragePooling2D()(X)
            elif unit_type == 'dense':
                layer_count += 1
                X = Dense(param_dict['num_units'], activation = param_dict['activation'], kernel_initializer='he_normal', name = stage_name_base_str + param_dict['name'])(X)
    
    model = Model(inputs = X_input, outputs = X, name='ResNet' + str(layer_count))
    name = 'ResNet' + str(layer_count)
    return model, name
    
