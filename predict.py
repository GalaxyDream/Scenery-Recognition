from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from trainingData import load_trainingdata
from testingData import load_testingdata
from keras.models import model_from_json
import os
from PIL import Image
import numpy as np
from resizeimage import resizeimage

print('loading data...')
data = np.empty((13,1,256,256),dtype="float32")
label = np.empty((13,),dtype="float32")
imgs = os.listdir("/home/xinsongdu/Desktop/bigdata/project/data/testImages2/")
num = len(imgs)
for i in range(num):
	img = Image.open("/home/xinsongdu/Desktop/bigdata/project/data/testImages2/"+imgs[i])
	img = img.convert('L')
	img = resizeimage.resize_cover(img, [256,256])
	arr = np.asarray(img,dtype="float32")
	data[i,0,:,:] = arr
	label[i] = int(imgs[i].split('_')[0])
	
scale = np.max(data)
data /= scale
mean = np.std(data)
data -= mean
print('loading model...')
model = model_from_json(open('my_model_architecture2.json').read())
model.load_weights('my_model_weights2.h5')
pdct=model.predict_classes(data, batch_size=1, verbose=1)
pdct2=model.predict_proba(data, batch_size=1, verbose=1)

print(pdct)
print(pdct2)
print(label)
print(pdct-label)







