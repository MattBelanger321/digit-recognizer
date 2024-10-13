import tensorflow as tf

import data_parser as parser

(images, labels) = parser.parse_mnist_data()
print(images[0].shape)


