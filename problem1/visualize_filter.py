from keras.models import load_model
import keras
from keras.models import Model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
#problem1(1)
model = load_model('cnn_model_new.hdf5') 
def conv_output(model, layer_name, img):
    """Get the output of conv layer.

    Args:
           model: keras model.
           layer_name: name of layer in the model.
           img: processed input image.

    Returns:
           intermediate_output: feature map.
    """
    # this is the placeholder for the input images
    input_img = model.input

    try:
        # this is the placeholder for the conv output
        out_conv = model.get_layer(layer_name).output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))

    # get the intermediate layer model
    intermediate_layer_model = Model(inputs=input_img, outputs=out_conv)

    # get the output of intermediate layer model
    intermediate_output = intermediate_layer_model.predict(img)

    return intermediate_output[0]





batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

            

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



layer_name = 'Conv2D_1' #获取层的名称
Conv2D_1 = Model(inputs=model.input, 
                                 outputs=model.get_layer(layer_name).output)#创建的新模型

filters,biases =Conv2D_1.get_weights()
f_min, f_max=filters.min(),filters.max()
filters=(filters -f_min)/(f_max-f_min)
print(Conv2D_1.name,filters.shape)
n_filters,ix=6,1
for i in range(n_filters):
    f=filters[:,:,:,i]
    for j in range(1):
        ax=plt.subplot(n_filters,3,ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[:,:,j],cmap='gray')
        ix+=1
plt.savefig('firstConv2D_1(6)')
plt.show()







layer_name2 = 'Conv2D_2' #获取层的名称
Conv2D_2 = Model(inputs=model.input, 
                                 outputs=model.get_layer(layer_name2).output)#创建的新模型

filters2,biases,p,q =Conv2D_2.get_weights()
f_min, f_max=filters2.min(),filters2.max()
filters2=(filters2 -f_min)/(f_max-f_min)
print(Conv2D_2.name,filters2.shape)
n_filters2,ix=6,1
for i in range(n_filters2):
    f=filters[:,:,:,i]
    for j in range(1):
        ax=plt.subplot(n_filters2,3,ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[:,:,j],cmap='gray')
        ix+=1
plt.savefig('firstConv2D_2(6)')
plt.show()


layer_name3 = 'Conv2D_3' #获取层的名称
Conv2D_3 = Model(inputs=model.input, 
                                 outputs=model.get_layer(layer_name3).output)#创建的新模型

filters3 =Conv2D_3.get_weights()[0]
f_min, f_max=filters3.min(),filters3.max()
filters3=(filters3 -f_min)/(f_max-f_min)
print(Conv2D_3.name,filters3.shape)
n_filters3,ix=6,1
for i in range(n_filters3):
    f=filters[:,:,:,i]
    for j in range(1):
        ax=plt.subplot(n_filters3,3,ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[:,:,j],cmap='gray')
        ix+=1
plt.savefig('firstConv2D_3(6)')
plt.show()


