import cv2
import numpy as np
import csv

# Path to the CSV file
csv_file_path = './data/train.csv'

# Open the CSV file
with open(csv_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the first line (header or unwanted row)
    
    # Iterate through each line starting from the second
    for idx, data_row in enumerate(reader, start=1):  # start=2 to reflect the actual line number
        # Discard the first column and convert the rest to integers
        array_784 = np.array([int(i) for i in data_row[1:]], dtype=np.uint8)
        
        # Reshape the array to 28x28
        image_cv2 = array_784.reshape((28, 28))
        
        # Save the image using OpenCV with naming convention sample_{idx}.png
        image_path_cv2 = f'./pngs/sample_{idx}.png'
        cv2.imwrite(image_path_cv2, image_cv2)
        
        print(f'Saved image: {image_path_cv2}')
