import os

imgs=os.listdir("/home/xinsongdu/Desktop/bigdata/project/data/bridge/")
for i in range(3021):
	os.rename('/home/xinsongdu/Desktop/bigdata/project/data/bridge/'+imgs[i],'/home/xinsongdu/Desktop/bigdata/project/data/testSet/'+imgs[i])