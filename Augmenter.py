dataset_path = "monkeypox/"
resultsPath = "results/"
#testCase="lastLayerFANDefault"
epochs = 200
testCase="withCNNAllDroputLastFAN"
# depicted from
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import datetime
import pickle
import numpy as np

from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt


# 1. Load Dataset
batch_size = 32
img_size = (224, 224)  # Smaller size to train from scratch

# Create training and validation datasets
train_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# 2. Normalize the Data
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# 3. Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical",input_shape=(224, 224, 3)),
    layers.RandomRotation(0.25,"constant"),
    layers.RandomZoom(0.2)
])

# Apply data augmentation to the training dataset
augmented_train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y)
)

# Check the number of batches and total images
num_train_batches = tf.data.experimental.cardinality(train_dataset).numpy()
num_val_batches = tf.data.experimental.cardinality(val_dataset).numpy()
num_train_images = num_train_batches * batch_size
num_val_images = num_val_batches * batch_size

print(f"Number of batches in training dataset: {num_train_batches}")
print(f"Total number of training images: {num_train_images}")
print(f"Number of batches in validation dataset: {num_val_batches}")
print(f"Total number of validation images: {num_val_images}")

# Display a random augmented image
for images, labels in augmented_train_dataset.take(1):
    random_index = tf.random.uniform([], minval=0, maxval=batch_size, dtype=tf.int32)
    random_image = images[random_index].numpy()
    random_label = labels[random_index].numpy()

    plt.imshow(random_image)
    plt.title(f"Label: {random_label}")
    plt.axis("off")
    plt.show()

