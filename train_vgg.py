# USAGE
# python train_vgg.py --dataset Output_melSpectrogram_3sec --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle --plot output/smallvggnet_plot.png

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch.smallvggnet import SmallVGGNet
from pyimagesearch.split_train_val import Split_train_val
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
	help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

#create separate folders for train and test
print("[INFO] Spliting files into train and test...")
Split_train_val.test_train_split(args["dataset"],0.10)

# initialize the data and labels
print("[INFO] loading images...")
#data = []
#labels = []
score=[]
trainX=[]
trainY=[]
testX=[]
testY=[]

# grab the image paths and randomly shuffle them
#imagePaths = sorted(list(paths.list_images(args["dataset"])))
#random.seed(42)
#random.shuffle(imagePaths)

# loop over the input images
#for imagePath in imagePaths:
	# load the image, resize it to 64x64 pixels (the required input
	# spatial dimensions of SmallVGGNet), and store the image in the
	# data list
#	image = cv2.imread(imagePath)
#	image = cv2.resize(image, (64, 64))
#	data.append(image)

	# extract the class label from the image path and update the
	# labels list
#	label = imagePath.split(os.path.sep)[-2]
#	labels.append(label)
	
	
	
# grab the training image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images('training_data')))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, resize it to 64x64 pixels (the required input
	# spatial dimensions of SmallVGGNet), and store the image in the
	# data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (300, 300))
	trainX.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	trainY.append(label)



	
# grab the testing image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images('testing_data')))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, resize it to 64x64 pixels (the required input
	# spatial dimensions of SmallVGGNet), and store the image in the
	# data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (300, 300))
	testX.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	testY.append(label)

	
	
	
# scale the raw pixel intensities to the range [0, 1]
#data = np.array(data, dtype="float") / 255.0
#labels = np.array(labels)
trainX = np.array(trainX, dtype="float") / 255.0
trainY = np.array(trainY)
testX = np.array(testX, dtype="float") / 255.0
testY = np.array(testY)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
#(trainX, testX, trainY, testY) = train_test_split(data,
#	labels, test_size=0.25, random_state=42)
#train_data_dir = './training_data'
#test_data_dir = './testing_data'
#nb_train_samples = 900
#nb_test_samples = 100

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize our VGG-like Convolutional Neural Network
model = SmallVGGNet.build(width=300, height=300, depth=3,
	classes=len(lb.classes_))

# initialize our initial learning rate, # of epochs to train for,
# and batch size
INIT_LR = 0.01
EPOCHS = 100
BS = 32
#batch_size = 32

# initialize the model and optimizer (you'll want to use
# binary_crossentropy for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))
print(confusion_matrix(testY.argmax(axis=1),predictions.argmax(axis=1)))
print(accuracy_score(testY.argmax(axis=1),predictions.argmax(axis=1)))
score = model.evaluate_generator(aug.flow(testX, testY, batch_size=BS),steps = len(testX/BS))
print(" Total: ", len(testX))
print("Loss: ", score[0], "Accuracy: ", score[1])



# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()                                                       
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (SmallVGGNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()