from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Conv3D, MaxPooling3D
import mlflow
import numpy as np
import config

def add_layer(net, *args, **kwargs):
    net.add(*args,**kwargs)
    print("after adding layer, output_shape=",net.output_shape)

def create():
    net = Sequential()#add model layers
    first = True
    for i in range(config.conv_layers):

        if config.max_pool_first and first is True:
            kwargs = dict()
            kwargs['pool_size']=(
                config.max_pool_first_size_t, 
                config.max_pool_first_size_xy, 
                config.max_pool_first_size_xy
            )
            kwargs['input_shape'] = (config.frames_per_point,1080,1920,3)
            add_layer(net,MaxPooling3D(**kwargs))
            first = False
        if config.conv_separable:
            kernel_shapes = (
                np.identity(3)
                *(config.kernel_size-1) 
                + np.ones((3,3))
            ).astype('int32')
        else:
            kernel_shapes = [config.kernel_size]
        for ks_i, curr_kernel_shape in enumerate(kernel_shapes):
            print("curr_kernel_shape",curr_kernel_shape)
            kwargs = dict()
            if first is True:
                kwargs['input_shape'] = (config.frames_per_point,1080,1920,3)
                first = False
            kwargs['kernel_size']=curr_kernel_shape
            kwargs['activation']=config.conv_activation
            if ks_i == 2: # last dimension of the kernel
                kwargs['strides']=(
                    config.stride_t,
                    config.stride_xy,
                    config.stride_xy
                )
            
            add_layer(net,Conv3D(config.conv_features, **kwargs))
        add_layer(net,MaxPooling3D(pool_size=(
            config.max_pool_size_t, 
            config.max_pool_size_xy, 
            config.max_pool_size_xy
        )))
    add_layer(net,Flatten())
    for i in range(config.dense_layers):
        size = int(config.first_dense_layer/(i+1))
        add_layer(net,Dense(size, activation=config.dense_activation))
    add_layer(net,Dense(2, activation='softmax'))


    net.compile(loss='categorical_crossentropy',
                  optimizer=config.optimizer,
                  metrics=['accuracy'])
    mlflow.log_param("n_params",net.count_params())
    net.summary()
    return net
