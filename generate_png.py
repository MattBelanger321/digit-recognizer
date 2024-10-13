import cv2

from PIL import Image

import data_parser as parser

(images, labels) = parser.parse_mnist_data()
print(images[0].shape)

for idx, image in enumerate(images, start=1):
	# Create a PIL Image from the array
    Image.fromarray(image, mode='L').save(f'./pngs/train_sample{idx}.png')  # 'L' mode for greyscale




