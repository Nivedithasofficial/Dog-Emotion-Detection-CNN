import tensorflow as tf
import cv2
import numpy as np

MODEL_PATH='models/animal_emotion_model.keras'
model=tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
# Add preprocessing and prediction code here.
