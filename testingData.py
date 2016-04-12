import os
from PIL import Image
import numpy as np

def load_testingdata():
	data = np.empty((18126,1,256,256),dtype="float32")
	label = np.empty((18126,),dtype="float32")
	imgs = os.listdir("/home/xinsongdu/Desktop/bigdata/project/data/testSet/")
	num = len(imgs)
	for i in range(num):
		img = Image.open("/home/xinsongdu/Desktop/bigdata/project/data/testSet/"+imgs[i])
		img = img.convert('L')
		arr = np.asarray(img,dtype="float32")
		data[i,0,:,:] = arr
		label[i] = int(imgs[i].split('_')[0])

	scale = np.max(data)
	data /= scale
	mean = np.std(data)
	data -= mean
	return data,label
