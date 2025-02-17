import tensorflow as tf
import os
import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
dataset_path = "monkeypox"
output_path = "monkeypox_aug"
img_size = (224, 224)

# 1. Count images per class
class_counts = {}
total_images = 0
for class_name in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):
        class_counts[class_name] = len(os.listdir(class_dir))
        total_images += class_counts[class_name]

# Determine total target size (10× original dataset size)
target_size = total_images * 10

# Calculate required augmentation per class
augmentation_ratios = {class_name: (target_size * (count / total_images)) - count for class_name, count in class_counts.items()}

# 2. Define Data Augmentation Generator
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create output dataset directory
os.makedirs(output_path, exist_ok=True)

for class_name, original_count in class_counts.items():
    class_dir = os.path.join(dataset_path, class_name)
    output_class_dir = os.path.join(output_path, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    # Copy original images
    for img_name in os.listdir(class_dir):
        shutil.copy(os.path.join(class_dir, img_name), output_class_dir)

    # Augment to reach 10× target
    num_augments = int(augmentation_ratios[class_name])
    images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]

    # Load images
    img_array = np.array([tf.keras.preprocessing.image.img_to_array(
        tf.keras.preprocessing.image.load_img(img, target_size=img_size)
    ) for img in images])

    # Generate new images
    generated = 0
    for x_batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_class_dir, save_prefix="aug", save_format="jpeg"):
        generated += 1
        if generated >= num_augments:
            break  # Stop when required images are created

print("Dataset augmented 10× while preserving class distribution!")
