import pandas as pd

import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l1
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2
import os
import matplotlib.pyplot as plt
import random
import numpy as np
from pathlib import Path
from keras.preprocessing.image import load_img
import cv2
# encode both columns label and variety
from sklearn.preprocessing import LabelEncoder
# ignore warnings
import warnings
warnings.filterwarnings('ignore')

train_datagen = ImageDataGenerator(rescale = 1./255,
                             rotation_range=40,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True,
                             validation_split=0.2)

val_datagen = ImageDataGenerator(rescale = 1./255,
                                validation_split=0.2)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Load The Train Images And Test Images

train_ds = train_datagen.flow_from_directory(
    directory = '/kaggle/input/fruit-classification10-class/MY_data/train',
    batch_size = 32,
    target_size = (224, 224),
    class_mode='categorical',
    subset="training",
    seed=123
)

validation_ds = val_datagen.flow_from_directory(
    directory='/kaggle/input/fruit-classification10-class/MY_data/train',
    batch_size=32,
    target_size=(224, 224),
    class_mode='categorical',
    subset="validation",
    seed=123
)


test_ds = train_datagen.flow_from_directory(
    directory = '/kaggle/input/fruit-classification10-class/MY_data/test',
    batch_size = 32,
    target_size = (224, 224),
    class_mode='categorical'
)

# Visualizing The Train Images

def visualize_images(path, num_images=5):

    # Get a list of image filenames
    image_filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    if not image_filenames:
        raise ValueError("No images found in the specified path")

    # Select random images
    selected_images = random.sample(image_filenames, min(num_images, len(image_filenames)))

    # Create a figure and axes
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3), facecolor='white')

    # Display each image
    for i, image_filename in enumerate(selected_images):
        # Load image
        image_path = os.path.join(path, image_filename)
        image = plt.imread(image_path)

        # Display image
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(image_filename)  # Set image filename as title

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

# Apple Images
# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/fruit-classification10-class/MY_data/train/Apple"

# Visualize 5 random images
visualize_images(path_to_visualize, num_images=5)

# Banana Images
# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/fruit-classification10-class/MY_data/train/Banana"

# Visualize 5 random images
visualize_images(path_to_visualize, num_images=5)

# Avocado Images
# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/fruit-classification10-class/MY_data/train/avocado"

# Visualize 5 random images
visualize_images(path_to_visualize, num_images=5)

# Cherry Images
# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/fruit-classification10-class/MY_data/train/cherry"

# Visualize 5 random images
visualize_images(path_to_visualize, num_images=5)

# Kiwi Images
# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/fruit-classification10-class/MY_data/train/kiwi"

# Visualize 5 random images
visualize_images(path_to_visualize, num_images=5)

# Mango Images
# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/fruit-classification10-class/MY_data/train/mango"

# Visualize 5 random images
visualize_images(path_to_visualize, num_images=5)

# Orange Images
# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/fruit-classification10-class/MY_data/train/orange"

# Visualize 5 random images
visualize_images(path_to_visualize, num_images=5)

# Pineapple Images
# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/fruit-classification10-class/MY_data/train/pinenapple"

# Visualize 5 random images
visualize_images(path_to_visualize, num_images=5)

# Strawberries Images
# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/fruit-classification10-class/MY_data/train/strawberries"

# Visualize 5 random images
visualize_images(path_to_visualize, num_images=5)

# Watermelon
# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/fruit-classification10-class/MY_data/train/watermelon"

# Visualize 5 random images
visualize_images(path_to_visualize, num_images=5)

# Model Building
# Load the pre-trained EfficientNetB4 model without the top classification layer
MobileNetV2_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3),
                              pooling='avg')

# Freeze the pre-trained base model layers
MobileNetV2_base.trainable = False

# Build the model
model = Sequential()

# Add the pre-trained Xception base
model.add(MobileNetV2_base)

# Batch Normalization
model.add(BatchNormalization())

# Dropout Layer
model.add(Dropout(0.35))

# Add a dense layer with 120 units and ReLU activation function
model.add(Dense(220, activation='relu'))

# Add a dense layer with 120 units and ReLU activation function
model.add(Dense(60, activation='relu'))

# Add the output layer with 1 unit and sigmoid activation function for binary classification
model.add(Dense(10, activation='softmax'))

# Check The Summary Of Model
model.summary()

# Compile The Model
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
 loss='categorical_crossentropy', metrics=['accuracy'])

# Model Training
%%time
# # Define the callback function
early_stopping = EarlyStopping(patience=10)

history= model.fit(train_ds,
    validation_data=validation_ds,
    steps_per_epoch=len(train_ds),
    epochs=100,
    callbacks=[early_stopping]
)

# Plotting The Loss And Accuracy
# evaluate the model
loss = model.evaluate(validation_ds)

# Plotting the training and testing loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# plot the accuracy of training and validation

# Plotting the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()
15/15 [==============================] - 1s 61ms/step - loss: 0.1788 - accuracy: 0.9522

# Predictions
# Get the class labels
class_labels = list(test_ds.class_indices.keys())

# Predict on each image and plot results
num_images = 20
num_images_per_row = 5  # Set the number of images per row
num_rows = 4

plt.figure(figsize=(15, 10))
for i in range(num_images):
    image, label = next(test_ds)
    predictions = model.predict(image)

    # Iterate over each image in the batch
    for j in range(len(image)):
        if i * len(image) + j < num_images:  # Check if the total number of images exceeds the desired count
            predicted_class = class_labels[np.argmax(predictions[j])]
            true_class = class_labels[np.argmax(label[j])]

            plt.subplot(num_rows, num_images_per_row, i * len(image) + j + 1)
            plt.imshow(image[j])
            plt.title(f'True: {true_class}\nPredicted: {predicted_class}')
            plt.axis('off')

plt.tight_layout()
plt.show()
