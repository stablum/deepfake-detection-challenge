from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Conv3D

import config

def add_layer(net, *args, **kwargs):
    net.add(*args,**kwargs)
    print("after adding layer, output_shape=",net.output_shape)

def create():
    net = Sequential()#add model layers
    for i in range(config.conv_layers):
        kwargs = dict(
            kernel_size=config.kernel_size, 
            strides=(
                config.stride_t,
                config.stride_xy,
                config.stride_xy
            ), 
            activation=config.conv_activation
        )
        if i == 0:
            kwargs['input_shape'] = (config.frames_per_point,1080,1920,3)
        add_layer(net,Conv3D(config.conv_features, **kwargs))
    add_layer(net,Flatten())
    for i in range(config.dense_layers):
        size = int(config.first_dense_layer/(i+1))
        add_layer(net,Dense(size, activation=config.dense_activation))
    add_layer(net,Dense(2, activation='softmax'))


    net.compile(loss='categorical_crossentropy',
                  optimizer=config.optimizer,
                  metrics=['accuracy'])
    return net
