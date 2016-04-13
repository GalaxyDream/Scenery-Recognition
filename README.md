# Scenery-Recognition

Scenery Recgnition with Python + Keras:

1. Introduction of Keras:
	Keras is a kind of deep learning library which is based on Theano. It refers to Torch when designing and created by Python. It is a highly-modularized neural network library and GPU and CPU are supported by it.
2. Data Access:
	We access the data through http://places.csail.mit.edu/user/. Then, we downloaded the dataset and 6 categories of scenery are included in the dataset. The total size of the dataset is 4G. Each category has 15100 images and all of them are 256*256 pixels.
3. Data Preprocessing(rename.py,move.py):
	After getting the data, we found that they are not so uniform. Some of them are RGB images while some others are gray-level images. Thus, we first transform all the images to gray-level images, which means their dimension are 256\*256\*1; Then, we change the file name of these images to the format of “CategoryID_ImageID”. There are 6 categories and each of them has an ID. Alley is 0, Amusement Park is 1, Aquarium is 2, bedroom is 3, bookstore is 4 and bridge is 5. Each of these categories has 15100 samples so ImageID is from 0 to 15099. For example, the 5th image of bedroom may be named “2_4”. The reason we do so is that it will be easy for us to extract label from these file names.
	After renaming the images, we choose 80% of each category as training data and put all of them into a directory named “trainingData” while put the rest into a directory named “testingData”.
4. Data Loading(trainingData.py, testingData.py):
	We wrote a data to load all these training data and testing data into our input: X_train, X_test. And we also extract labels from the names of files and stored them to Y_train, Y_test.
5. Data Normalization:
	Because the span of the image data matric is from 0 to 255, we then normalize them to a number from 0 to 1.
6. Architecture Design(fit.py):
	The input dimension is 256*256*1 and the first convolution layer has a kernel size of 3*3 and kernel number of 32. The second convolution layer has a kernel size of 3*3 and kernel number of 32 and with a 2*2 maxpooling layer. The MaxPooling layer has a dropout of 0.25. The third convolution layer has a kernel size of 3*3 and kernel number of 64. The fourth convolution layer has a kernel size of  3*3 and kernel number of 32 and with a 2*2 maxpooling layer. The MaxPooling layer has a dropout of 0.25. Then, a fully connected layer with 256 neurons follows the fourth convolution layer. The last layer is fully connected layer with 6 neurons and 0.5 dropout. The activation function of the last layer is Softmax while that of other layers are Relu, which can speed up the training process.

7. Training(fit.py):
	For hyperparameters, we set learning rate to 0.1, decay to 1e-6, momentum to 0.9 and optimizer to SGD. After 20 epochs training, we get a loss of 0.0463 and an accuracy of 0.729.

8. Prediction(predict.py):
	We first randomly choose 12 256*256 images from the Internet, then use our saved model to predict. 11 of these 12 images are correctly recognized by the computer, the accuracy is about 91.7%. Then, we again choose 13 images with random size and input them to our model, 7 of them are correctly recognized by the computer, the accuracy is about 53.8%. The accuracy of randomly guess is about 33%, which is much lower than that of our model. Besides, it seems that our model works better for images with a size of 256*256.
