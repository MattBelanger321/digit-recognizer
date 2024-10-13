import cv2

from PIL import Image

import data_parser as parser


# generate training images
(images, labels) = parser.parse_mnist_training_data()
print(images[0].shape)

for idx, image in enumerate(images, start=1):
	# Create a PIL Image from the array
    Image.fromarray(image, mode='L').save(f'./train_pngs/train_sample{idx}.png')  # 'L' mode for greyscale

# generate test images
(images, labels) = parser.parse_mnist_test_data()
print(images[0].shape)

for idx, image in enumerate(images, start=1):
	# Create a PIL Image from the array
    Image.fromarray(image, mode='L').save(f'./test_pngs/test_sample{idx}.png')  # 'L' mode for greyscale




