import os

imgs=os.listdir("/home/xinsongdu/Desktop/bigdata/project/data/bridge/")
for i in range(len(imgs)):
	os.rename(imgs[i],'6_'+str(i))