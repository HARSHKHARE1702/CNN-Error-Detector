
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (128, 128)
MODEL_PATH = "error_cnn.keras"

model = tf.keras.models.load_model(MODEL_PATH)

def predict_screenshot(image_path):
    # Load and preprocess screenshot
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # Model includes pixel normalization
    probability_normal = float(
        model.predict(image_array, verbose=0)[0][0]
    )

    probability_error = 1.0 - probability_normal

    if probability_error >= 0.5:
        label = "ERROR DETECTED"
        confidence = probability_error
    else:
        label = "NORMAL SCREEN"
        confidence = probability_normal

    print(f"Screenshot: {image_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Error probability: {probability_error:.2%}")

    return label, confidence


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py screenshot.png")
        sys.exit(1)

    predict_screenshot(sys.argv[1])
