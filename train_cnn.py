import tensorflow as tf

import data_parser as parser

from numpy import mean
from numpy import std
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD

# define cnn model
def define_model():
	model = Sequential()
	# Define the input layer
	model.add(Input(shape=(28, 28, 1)))  # Assuming 28x28 grayscale images for digit recognition
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# Normalize 0-1 a single image
def normalize(image, scale=255.0):
	# convert from integers to floats
	image_norm = image.astype('float32')
	# normalize to range 0-1
	image_norm = image_norm / scale
	# return normalized images
	return image_norm

# normailze the entire training data
def normalize_images(images_raw):
	images = []
	for i in range(images_raw.shape[0]):
		images.append( normalize(images_raw[i]) )
	return images

def interpret_one_hot(one_hot, delta):
	for index, value in enumerate(one_hot):
		if abs(value - 1) < delta:
			return index  # Return the index of the first element found
	return -1  # Return -1 if no such element is found

def train_model():
	(images_raw, labels) = parser.parse_mnist_training_data()
	images = normalize_images(images_raw)	# [0,1]

	images = np.array(images)
	labels = np.array(labels)

	model = define_model()


	# one hot encode target values
	labels = to_categorical(labels)

	# define model
	model = define_model()
	# split labelled data into test and train data
	X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)

	# fit model
	history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)
	_, acc = model.evaluate(X_test, y_test, verbose=0)
	print('> %.3f' % (acc * 100.0))
	return model


model = train_model() # train the model

model.encode()

# run it against an unlabelled sample
(raw_kaggle_test, labels) = parser.parse_mnist_test_data()
kaggle_test = normalize_images(raw_kaggle_test)	# [0,1]
kaggle_test = np.array(kaggle_test)
# Reshape the first sample to (1, 28, 28, 1)
kaggle_test_reshaped = kaggle_test[0].reshape(1, 28, 28, 1)

print(kaggle_test_reshaped.shape)  # Should print (1, 28, 28, 1)
print(interpret_one_hot(model.predict(kaggle_test_reshaped)[0],0.001))