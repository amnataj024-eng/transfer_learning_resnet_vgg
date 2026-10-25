"""
ResNet50 Transfer Learning Assignment Script
Author: ML Engineering Mentor
Description: Trains a ResNet50 model using ImageNet weights on a custom dataset.
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- 1. Configuration & Hyperparameters ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FEATURE_EXTRACTION = 5
EPOCHS_FINE_TUNING = 5
NUM_CLASSES = 2  # Adjust based on your dataset (e.g., cats vs dogs = 2)
DATA_DIR = "dataset"

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "validation")
test_dir = os.path.join(DATA_DIR, "test")

# --- 2. Data Loading & Augmentation ---
print("[INFO] Loading training, validation, and test datasets...")

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
print(f"[INFO] Detected Classes: {class_names}")

# Data Augmentation pipeline to prevent overfitting
data_augmentation = models.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation",
)


# Apply preprocessing required for ResNet50
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

# --- 3. Build Model with ResNet50 Base ---
print("[INFO] Loading pre-trained ResNet50 base model...")
base_model = ResNet50(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)

# Freeze convolutional base layers
base_model.trainable = False

# Construct custom classification head
inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs, name="ResNet50_TransferLearning")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# --- 4. Phase 1: Train Classification Head ---
print("[INFO] Starting Phase 1: Training classifier head...")
history_head = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS_FEATURE_EXTRACTION
)

# --- 5. Phase 2: Fine-Tuning ---
print(
    "[INFO] Starting Phase 2: Fine-tuning top layers of the convolutional base..."
)
base_model.trainable = True

# Freeze earlier layers, unfreeze the last 30 layers
for layer in base_model.layers[:-30]:
  layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),  # Low learning rate for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_finetune = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS_FINE_TUNING
)

# --- 6. Evaluate & Save ---
print("[INFO] Evaluating model on test set...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"[RESULT] Test Accuracy (ResNet50): {test_acc * 100:.2f}%")

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

model_path = os.path.join("models", "resnet50_best.h5")
model.save(model_path)
print(f"[INFO] Model saved successfully to {model_path}")


# --- 7. Plot Learning Curves ---
def plot_history(h1, h2, model_name):
  acc = h1.history["accuracy"] + h2.history["accuracy"]
  val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
  loss = h1.history["loss"] + h2.history["loss"]
  val_loss = h1.history["val_loss"] + h2.history["val_loss"]

  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  plt.plot(acc, label="Train Accuracy")
  plt.plot(val_acc, label="Validation Accuracy")
  plt.title(f"{model_name} - Accuracy")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(loss, label="Train Loss")
  plt.plot(val_loss, label="Validation Loss")
  plt.title(f"{model_name} - Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()

  plt.savefig(f"outputs/{model_name.lower()}_training_plot.png")
  plt.close()


plot_history(history_head, history_finetune, "ResNet50")
print("[INFO] Training plots saved to outputs/")