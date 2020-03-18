from keras.models import load_model
import numpy as np
from PIL import Image, ImageDraw
model = load_model('cnn_model_new.hdf5')
impath = 'covered1.jpg'
img1=Image.open(impath)
img1=np.asarray(img1)/255.00
arr1 = np.array(img1).reshape((1, 28, 28, 1))

impath = 'covered2.jpg'
img2=Image.open(impath)
img2=np.asarray(img2)/255.00
arr2 = np.array(img2).reshape((1, 28, 28, 1))

impath = 'covered3.jpg'
img3=Image.open(impath)
img3=np.asarray(img3)/255.00
arr3 = np.array(img3).reshape((1, 28, 28, 1))

impath = 'covered4.jpg'
img4=Image.open(impath)
img4=np.asarray(img4)/255.00
arr4 = np.array(img4).reshape((1, 28, 28, 1))

impath = 'covered5.jpg'
img5=Image.open(impath)
img5=np.asarray(img5)/255.00
arr5 = np.array(img5).reshape((1, 28, 28, 1))

impath = 'covered6.jpg'
img6=Image.open(impath)
img6=np.asarray(img6)/255.00
arr6 = np.array(img6).reshape((1, 28, 28, 1))
pre1 = np.argmax(model.predict(arr1))
pre2 = np.argmax(model.predict(arr2))
pre3 = np.argmax(model.predict(arr3))
pre4 = np.argmax(model.predict(arr4))
pre5 = np.argmax(model.predict(arr5))
pre6 = np.argmax(model.predict(arr6))

print(pre1,pre2,pre3,pre4,pre5,pre6)
