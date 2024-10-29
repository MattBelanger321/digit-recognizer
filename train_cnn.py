import data_parser as parser

from numpy import mean, std
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
import pandas as pd

# Define CNN model
def define_model():
    model = Sequential()
    # Input shape specified in the first Conv2D layer
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # Compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model():
    # Load data from parser
    (images_raw, labels) = parser.parse_mnist_training_data()

    # Normalize images
    images = normalize_images(images_raw)

    # Convert labels to numpy array
    labels = np.array(labels)

    # One-hot encode target values
    labels = to_categorical(labels)

    # Define model
    model = define_model()

    # Split labeled data into test and train data
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)

    # Fit model and save history
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    
    # Save the training history to a CSV file
    save_training_history(history)

    # Plotting results
    plotResults(history)
    
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print('> %.3f' % (acc * 100.0))

    return model

# Normalize 0-1 a single image
def normalize(image, scale=255.0):
    # Convert from integers to floats and normalize to range 0-1
    return image.astype('float32') / scale

# Normalize the entire training data
def normalize_images(images_raw):
    return images_raw.astype('float32') / 255.0

# Function to interpret the one-hot encoded output
def interpret_one_hot(one_hot, delta):
    for index, value in enumerate(one_hot):
        if abs(value - 1) < delta:
            return index  # Return the index of the first element found
    return -1  # Return -1 if no such element is found

# Save training history for MATLAB
def save_training_history(history, filename='training_history.csv'):
    df = pd.DataFrame(history.history)
    df.to_csv(filename, index=False)
    return df

# This is a helper funtion to visualize the history metrics against epochs
def plotResults(history):
    # Fetch metrics
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(12, 6))

    # Subplot is used to plot the results together
    # Here, we plotting Training Loss in plot 1
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Here, we plotting Training Accuracy in plot 2
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r-', label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # For adjusting padding between and around the plots
    plt.tight_layout()
    plt.show()

# Call the function to train the model
model = train_model()

# Run it against an unlabeled sample
(raw_kaggle_test, labels) = parser.parse_mnist_test_data()
kaggle_test = normalize_images(raw_kaggle_test)  # [0,1]
kaggle_test = np.array(kaggle_test)

# Reshape the first sample to (1, 28, 28, 1)
kaggle_test_reshaped = kaggle_test[0].reshape(1, 28, 28, 1)

print(kaggle_test_reshaped.shape)  # Should print (1, 28, 28, 1)
print(interpret_one_hot(model.predict(kaggle_test_reshaped)[0], 0.001))
