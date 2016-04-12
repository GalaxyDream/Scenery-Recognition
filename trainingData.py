import os
from PIL import Image
import numpy as np

def load_trainingdata():
	data = np.empty((72474,1,256,256),dtype="float32")
	label = np.empty((72474,),dtype="uint8")
	imgs = os.listdir("/home/xinsongdu/Desktop/bigdata/project/data/trainingSet/")
	num = len(imgs)
	for i in range(num):
		img = Image.open("/home/xinsongdu/Desktop/bigdata/project/data/trainingSet/"+imgs[i])
		img = img.convert('L')
		arr = np.asarray(img,dtype="float32")
		data[i,0,:,:] = arr
		label[i] = int(imgs[i].split('_')[0])

	scale = np.max(data)
	data /= scale
	mean = np.std(data)
	data -= mean
	return data,label
