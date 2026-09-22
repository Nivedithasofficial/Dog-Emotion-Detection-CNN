import tensorflow as tf
import numpy as np
import cv2
import sys


MODEL_PATH = "models/animal_emotion_model_version4.keras"

IMG_HEIGHT = 224
IMG_WIDTH = 224

CLASS_NAMES = [
    "angry",
    "happy",
    "relaxed",
    "sad"
]

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


def predict_emotion(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Could not read image.")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    image = image.astype(np.float32)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image, verbose=0)

    predicted_index = np.argmax(predictions[0])

    predicted_emotion = CLASS_NAMES[predicted_index]

    confidence = predictions[0][predicted_index]

    print(f"Predicted emotion: {predicted_emotion}")
    print(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    predict_emotion(image_path)
