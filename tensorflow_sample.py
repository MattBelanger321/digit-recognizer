import tensorflow as tf

import data_parser as parser

from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD

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
	for i in range(images_raw.shape[0]):
		images.append( normalize(images_raw[i]) )
	return images

(images_raw, labels) = parser.parse_mnist_data()
images = normalize_images(images_raw)








