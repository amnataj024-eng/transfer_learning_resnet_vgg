"""
VGG16 Transfer Learning Assignment Script
Author: ML Engineering Mentor
Description: Trains a VGG16 model using ImageNet weights on a custom dataset.
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# --- 1. Configuration & Hyperparameters ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 6
NUM_CLASSES = 2
DATA_DIR = "dataset"

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "validation")
test_dir = os.path.join(DATA_DIR, "test")

# --- 2. Data Loading & Augmentation ---
print("[INFO] Loading datasets for VGG16...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

class_names = train_dataset.class_names

data_augmentation = models.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ],
    name="vgg_augmentation",
)


def preprocess_train(images, labels):
  images = data_augmentation(images, training=True)
  images = preprocess_input(images)
  return images, labels


def preprocess_val_test(images, labels):
  images = preprocess_input(images)
  return images, labels


train_ds = train_dataset.map(
    preprocess_train, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)
val_ds = val_dataset.map(
    preprocess_val_test, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)
test_ds = test_dataset.map(
    preprocess_val_test, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# --- 3. Build Model with VGG16 Base ---
print("[INFO] Loading pre-trained VGG16 base model...")
base_model = VGG16(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.Flatten()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs, name="VGG16_TransferLearning")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# --- 4. Training ---
print("[INFO] Training VGG16 model...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# --- 5. Evaluate & Save ---
test_loss, test_acc = model.evaluate(test_ds)
print(f"[RESULT] Test Accuracy (VGG16): {test_acc * 100:.2f}%")

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

model_path = os.path.join("models", "vgg16_best.h5")
model.save(model_path)
print(f"[INFO] Model saved successfully to {model_path}")

# --- 6. Plot Learning Curves ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("VGG16 - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("VGG16 - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("outputs/vgg16_training_plot.png")
plt.close()
print("[INFO] Training plots saved to outputs/")