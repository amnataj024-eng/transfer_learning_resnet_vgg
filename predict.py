"""
Inference Script for Image Classification
Description: Loads a trained ResNet50 or VGG16 model and predicts the class of a new image.
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Parse command line arguments
parser = argparse.ArgumentParser(description="Classify an image using saved model")
parser.add_argument(
    "--model",
    type=str,
    default="models/resnet50_best.h5",
    help="Path to trained model (.h5)",
)
parser.add_argument(
    "--image", type=str, required=True, help="Path to input image"
)
args = parser.parse_args()

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["cats", "dogs"]  # Update according to your folder labels

print(f"[INFO] Loading model from {args.model}...")
model = tf.keras.models.load_model(args.model)

print(f"[INFO] Processing image: {args.image}...")
img = image.load_img(args.image, target_size=IMAGE_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Preprocessing depends on the model architecture selected
if "resnet" in args.model.lower():
  from tensorflow.keras.applications.resnet50 import preprocess_input
else:
  from tensorflow.keras.applications.vgg16 import preprocess_input

img_array = preprocess_input(img_array)

print("[INFO] Performing prediction...")
predictions = model.predict(img_array)
predicted_class_idx = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100

print("\n--- Prediction Results ---")
print(f"Predicted Class: {CLASS_NAMES[predicted_class_idx].upper()}")
print(f"Confidence: {confidence:.2f}%")
print("--------------------------\n")