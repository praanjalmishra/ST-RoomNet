import os
from xml.etree import ElementTree
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from spatial_transformer import ProjectiveTransformer, AffineTransformer
#from tensorflow.keras.applications.xception import preprocess_input, Xception
from tensorflow.keras.applications.convnext import ConvNeXtTiny, preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import *
from scipy.interpolate import interp1d
import random
import scipy.io
import math
from scipy import ndimage
from sklearn.utils import shuffle


val_path = '/home/scai/mr/surface_relabel/train/'

img = cv2.imread(val_path+'0a578e8af1642d0c1e715aaa04478858ac0aab01.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (400 , 400))
img1 =  np.array(img,copy=True)
img = img[tf.newaxis,...]
img = preprocess_input(img)

seg = cv2.imread(val_path+'0a578e8af1642d0c1e715aaa04478858ac0aab01.jpg',0)
seg = cv2.resize(seg, (400 ,400), interpolation= cv2.INTER_NEAREST)
seg = seg/51.0
 
ref_img = tf.io.read_file('ref_img2.png')
ref_img = tf.io.decode_png(ref_img)
ref_img = tf.cast(ref_img, tf.float32) / 51.0
ref_img = ref_img[tf.newaxis,...]
#ref_img = tf.tile(ref_img, [1,1,1,1])
print(ref_img.shape)


base_model = ConvNeXtTiny(include_top=False, weights="imagenet", input_shape= (400,400,3), pooling = 'avg')
theta = Dense(8)(base_model.output)
stl= ProjectiveTransformer((400,400)).transform(ref_img, theta)
model = Model(base_model.input, stl)


#model.summary()
model.load_weights('/home/scai/mr/ST-RoomNet/Weight_ST_RroomNet_ConvNext.h5')


out= model.predict(img)

out = np.rint(out[0,:,:,0])



plt.figure('seg')
plt.imshow(out, vmin=1, vmax=5)
plt.axis('off')  # Turn off the axis
plt.savefig('output.png', bbox_inches='tight', pad_inches=0)  # Save the figure without borders


plt.figure('gt')
plt.imshow(seg , vmin = 1, vmax= 5)
plt.figure('img')
plt.imshow(img1)
plt.show()


segmented_image_path = 'output.png'
segmented_image = cv2.imread(segmented_image_path, cv2.IMREAD_GRAYSCALE)

blurred_image = cv2.GaussianBlur(segmented_image, (5, 5), 0)

edges = cv2.Canny(blurred_image, 50, 150)

kernel = np.ones((3, 3), np.uint8)  # kernel for dilation
enhanced_edges = cv2.dilate(edges, kernel, iterations=1)

overlay = cv2.addWeighted(segmented_image, 0.8, enhanced_edges, 1, 0)

# Display the results
plt.figure('Canny Edge Detection')
plt.title('Original Segmented Image')
plt.imshow(segmented_image, cmap='gray')
plt.axis('off')

plt.title('Detected Edges')
plt.imshow(enhanced_edges, cmap='gray')
plt.axis('off')
plt.show()


plt.title('Overlay on Segmented Image')
plt.imshow(overlay, cmap='gray')
plt.axis('off')

plt.show()
