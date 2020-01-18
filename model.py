from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Conv3D

def add_layer(net, *args, **kwargs):
    net.add(*args,**kwargs)
    print("after adding layer, output_shape=",net.output_shape)

def create():
    net = Sequential()#add model layers
    add_layer(net,Conv3D(5, kernel_size=7, strides=(2,7,7), activation='elu', input_shape=(50,1080,1920,3)))
    add_layer(net,Conv3D(5, kernel_size=7, strides=(2,7,7), activation='elu'))
    add_layer(net,Conv3D(5, kernel_size=7, strides=(2,7,7), activation='elu'))
    add_layer(net,Flatten())
    add_layer(net,Dense(10, activation='sigmoid'))
    add_layer(net,Dense(2, activation='softmax'))


    net.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return net
