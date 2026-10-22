
import re
from pathlib import Path

import easyocr
import numpy as np
import tensorflow as tf
from PIL import Image


# ---------------- CONFIGURATION ----------------

MODEL_PATH = "error_cnn.keras"
IMAGE_SIZE = (128, 128)
ERROR_THRESHOLD = 0.50

# Your model was trained with:
# Error = 0
# Normal = 1


# ---------------- LOAD MODELS ----------------

print("Loading CNN model...")
cnn_model = tf.keras.models.load_model(MODEL_PATH)

print("Loading OCR model...")
ocr_reader = easyocr.Reader(["en"], gpu=False)

print("Models loaded successfully!\n")


# ---------------- CNN ERROR DETECTION ----------------

def detect_error(image_path):
    image = Image.open(image_path).convert("RGB")

    resized = image.resize(IMAGE_SIZE)
    image_array = np.array(resized, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Model output is probability of Normal (class 1).
    probability_normal = float(
        cnn_model.predict(image_array, verbose=0)[0][0]
    )

    probability_error = 1.0 - probability_normal

    prediction = (
        "Error"
        if probability_error >= ERROR_THRESHOLD
        else "Normal"
    )

    return image, prediction, probability_error


# ---------------- OCR TEXT EXTRACTION ----------------

ERROR_PATTERNS = [
    r"\btraceback\b",
    r"\bexception\b",
    r"\berror\b",
    r"\bfailed\b",
    r"\bfailure\b",
    r"\bcaused by\b",
    r"\bValueError\b",
    r"\bTypeError\b",
    r"\bKeyError\b",
    r"\bIndexError\b",
    r"\bAttributeError\b",
    r"\bNameError\b",
    r"\bImportError\b",
    r"\bModuleNotFoundError\b",
    r"\bRuntimeError\b",
    r"\bNullPointerException\b",
    r"\bNullReferenceException\b",
    r"\bSQLException\b",
    r"\bHTTP\s*[45]\d\d\b",
]

TRACE_PATTERNS = [
    r"^\s*Traceback\b",
    r"^\s*File\s+['\"]",
    r"^\s*at\s+[\w.$<>]+\(",
    r"^\s*Caused by:",
    r"^\s*Suppressed:",
    r"^\s*\.\.\.\s+\d+\s+more",
]


def looks_like_error(text):
    return any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in ERROR_PATTERNS
    )


def looks_like_stack_trace(text):
    return any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in TRACE_PATTERNS
    )


def extract_error_message(image):
    image_array = np.array(image.convert("RGB"))

    # Read text regions from the screenshot.
    results = ocr_reader.readtext(
        image_array,
        detail=1,
        paragraph=False
    )

    # Sort approximately from top to bottom, left to right.
    detected_lines = []

    for bbox, text, confidence in results:
        text = text.strip()

        if not text:
            continue

        top = min(point[1] for point in bbox)
        left = min(point[0] for point in bbox)

        detected_lines.append((top, left, text))

    detected_lines.sort(key=lambda item: (item[0], item[1]))
    lines = [item[2] for item in detected_lines]

    if not lines:
        return "", lines

    # Identify error messages and stack-trace lines.
    selected = set()

    for index, line in enumerate(lines):
        if looks_like_error(line):
            selected.add(index)

            # Include nearby context for wrapped messages.
            for nearby in range(
                max(0, index - 1),
                min(len(lines), index + 2)
            ):
                selected.add(nearby)

        if looks_like_stack_trace(line):
            selected.add(index)

    # Preserve lines after a Python traceback header.
    for index, line in enumerate(lines):
        if re.search(r"\bTraceback\b", line, re.IGNORECASE):
            for following in range(index, len(lines)):
                selected.add(following)

                # Stop after the likely final exception line.
                if (
                    following > index
                    and looks_like_error(lines[following])
                    and not looks_like_stack_trace(lines[following])
                ):
                    break

    # Keep selected lines in original reading order.
    extracted_lines = [lines[i] for i in sorted(selected)]

    # Remove duplicate OCR results.
    unique_lines = []
    seen = set()

    for line in extracted_lines:
        key = line.casefold()

        if key not in seen:
            seen.add(key)
            unique_lines.append(line)

    return "\n".join(unique_lines).strip(), lines


# ---------------- MAIN PROGRAM ----------------

def main():
    image_path = input(
        "Enter the screenshot path: "
    ).strip().strip('"')

    if not Path(image_path).is_file():
        print("\nScreenshot not found. Check the file path.")
        return

    try:
        image, prediction, error_probability = detect_error(
            image_path
        )

        print("\n========== CNN RESULT ==========")
        print(f"Prediction: {prediction}")
        print(f"Error probability: {error_probability:.2%}")

        if prediction == "Error":
            print("\n========== OCR RESULT ==========")

            error_text, all_text = extract_error_message(image)

            if error_text:
                print("Extracted error message / stack trace:\n")
                print(error_text)
            else:
                print(
                    "CNN detected an error, but no recognizable "
                    "error message or stack trace was found."
                )

                print("\nAll OCR text for debugging:")
                print("\n".join(all_text))

        else:
            print(
                "\nNo OCR extraction performed because the CNN "
                "classified this screenshot as Normal."
            )

    except Exception as exc:
        print(f"\nProcessing failed: {exc}")


if __name__ == "__main__":
    main()
