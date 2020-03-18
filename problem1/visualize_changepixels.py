from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np

import numpy as np
from PIL import Image

import cv2
import math
import numpy as np
from keras.models import load_model
#problem1 (3)
model=load_model("cnn_model_new.hdf5")

src = cv2.imread('1.jpg',0)
arr = np.array(src).reshape(1,28,28,1)
before_shift = model.predict(arr)

print('before_shift:',np.argmax(before_shift))


rows = src.shape[0]
cols = src.shape[1]
img_cut = src[3:25,0:28]

img =cv2.copyMakeBorder(img_cut,3,3,0,0,cv2.BORDER_CONSTANT, value=1)



arr1 = np.array(img).reshape((1, 28, 28, 1))






model = load_model("cnn_model_new.hdf5")

after_shift=model.predict(arr1)

print('after_shift:',np.argmax(after_shift))

