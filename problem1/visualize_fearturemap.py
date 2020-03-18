from __future__ import division, print_function, absolute_import
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np

import numpy as np
from PIL import Image

# 读取训练好的模型
model = load_model("cnn_model_new.hdf5")
img = Image.open('8.jpg').convert('L')

if img.size[0] != 28 or img.size[1] != 28:
    img = img.resize((28, 28))

arr = []

for i in range(28):
    for j in range(28):
        pixel = 1.0 - float(img.getpixel((j, i)))/255.0
        arr.append(pixel)

arr1 = np.array(arr).reshape((1, 28, 28, 1))





model = Model(inputs=model.inputs,outputs=model.layers[2].output)
model.summary()

feature_map=model.predict(arr1)
square =64
ix =1
for _ in range(square):
    ax=pyplot.subplot(8,8,ix)
    ax.set_xticks([])
    ax.set_yticks([])
    pyplot.imshow(feature_map[0,:,:,ix-1],cmap='gray')
    ix+=1
pyplot.savefig("8_afterConv")
pyplot.show()
