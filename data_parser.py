import numpy as np
import csv

# Helper function to read kaggle data

def parse_mnist_training_data(csv_file_path = './data/train.csv', width=28, height=28):
    # Open the CSV file
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        
        images = []  # Initialize an empty list to store images
        labels = []	# store the label for each image

        # Iterate through each line starting from the second
        for idx, data_row in enumerate(reader, start=1):  # start=2 to reflect the actual line number
            # Discard the first column and convert the rest to integers
            array_1d = np.array([int(i) for i in data_row[1:]], dtype=np.uint8)
            
            # Reshape the array to 28x28
            image = array_1d.reshape((width, height))

            # Add the reshaped image to the list
            images.append(image)
            labels.append(data_row[0])
    
    # Optionally convert the list to a NumPy array
    return [np.array(images), np.array(labels).astype(int)]

def parse_mnist_test_data(csv_file_path = './data/test.csv', width=28, height=28):
    # Open the CSV file
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        
        images = []  # Initialize an empty list to store images
        labels = []	# store the label for each image

        # Iterate through each line starting from the second
        for idx, data_row in enumerate(reader, start=1):  # start=2 to reflect the actual line number
            # Discard the first column and convert the rest to integers
            array_1d = np.array([int(i) for i in data_row], dtype=np.uint8)
            
            # Reshape the array to 28x28
            image = array_1d.reshape((width, height))

            # Add the reshaped image to the list
            images.append(image)
            labels.append(data_row[0])
    
    # Optionally convert the list to a NumPy array
    return [np.array(images), np.array(labels).astype(int)]
