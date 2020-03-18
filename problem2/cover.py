from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import pylab
import numpy as np
from keras.models import load_model
import cv2
impath = '6.jpg'
img1=Image.open(impath)
img1=np.asarray(img1)/255.00
picture_list =[]
for i in range(0,22,2):
    for j in range(0,22,2):

        img2=Image.open(impath)
        draw=ImageDraw.Draw(img2)
        
        draw.rectangle((j,i,j+8,i+8),fill='black')
        img2=np.asarray(img2)/255.00
        j=j+2
        arr1 = np.array(img2).reshape((1, 28, 28, 1))
        picture_list.append(arr1)
    i=i+2    
picture_list=np.array(picture_list)
#print(picture_list.shape)
#print(picture_list)



model = load_model('cnn_model_new.hdf5')
prob=[]
prob_map=[]
high_prob_map=[]
for i in range(121):

    prob.append(model.predict(picture_list[i]))#all probality

for i in range(121):

    prob_map.append((prob[i][0][6]))
    high_prob_map.append(max(prob[i][0]))
print ('the proba of 6:',prob_map)



print('the highest prob:',high_prob_map)

high_label_map=[]
for j in range(121):

    high_label_map.append(np.argmax(prob[j][0]))

print('highest proba--label:',high_label_map)

# we can see the 59th 60th 60th 71st 82nd 93rd, the predict label is 5.
#when i=12, j =6,8.  i=13,j=6,8.  i=14,j=8.  i=15,j=8.
img2=Image.open(impath)
draw=ImageDraw.Draw(img2)
 
draw.rectangle((6,12,6+8,12+8),fill='black')
img2=np.asarray(img2)/255.00
plt.imshow(img2)
plt.savefig('effective cover1')
plt.show()

img2=Image.open(impath)
draw=ImageDraw.Draw(img2)
 
draw.rectangle((8,12,8+8,12+8),fill='black')
img2=np.asarray(img2)/255.00
plt.imshow(img2)
plt.savefig('effective cover2')
plt.show()


img2=Image.open(impath)
draw=ImageDraw.Draw(img2)
 
draw.rectangle((6,13,6+8,13+8),fill='black')
img2=np.asarray(img2)/255.00
plt.imshow(img2)
plt.savefig('effective cover3')
plt.show()

img2=Image.open(impath)
draw=ImageDraw.Draw(img2)
 
draw.rectangle((8,12,8+8,12+8),fill='black')
img2=np.asarray(img2)/255.00
plt.imshow(img2)
plt.savefig('effective cover4')
plt.show()



draw.rectangle((8,15,8+8,15+8),fill='black')
img2=np.asarray(img2)/255.00
plt.imshow(img2)
plt.savefig('effective cover5')
plt.show()



draw.rectangle((8,14,8+8,14+8),fill='black')
img2=np.asarray(img2)/255.00
plt.imshow(img2)
plt.savefig('effective cover6')
plt.show()

