
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
#-----ISSUE CLASSIFICATION-------------
   ISSUE KEYWORD = {
       "Login Issue":{
           "login":3,
           "log in":3,
           "sign in":3,
           "signin":3,
           "authentication":3,
           "invalid credentials":5,
           "Incorrect password":5,
           "wrong passoword":5,
           "password mismatch":4,
           "unauthorized":3,
           "session expired":3,
           "login failed":5,
           "invalid username":4,
           "invalid user":2,
       },
       "Registration Issue":{
           "register":3,
           "registration":4,
           "singup":3,
           "sing up":3,
           "create account":4,
           "already exists":3,
           "user already exists":5,
           "duplicate user":5,
           "registration failed":5,
           "otp":2,
           "verification code":3,
           "email already exists":5,
           "mobile already registered":5,
       },
       "MPIN Issue":{
           "mpin":5,
           "m-pin":5,
           "m pin":5,
           "Incorrect pin":4,
           "invalid pin":4,
           "wrong pin":4,
           "pin verification":4,
           "pin mismatch":4,
           "reset pin":4,
           "set pin":3,
           "forgot pin":4,
           "mpin failed":5,
           "mpin verification":5,
           "transaction pin":2,
       },
   }
    def classify_Issue(error_text, all_ocr_text=""):
        combined_text = f"{all_ocr_text}\n{error_text}".lower()
        scores = {
            category=0
            for category in ISSUE_KEYWORDS
}
   matched_keywords={
                category:[]
                for category in ISSUE_KEYWORDS
            }
   for category, keywords in ISSUE_KEYWORDS.items():
       for keyword, weight in keyword.items():
           if keyword in combined_text:
               scores[category] += weight
               matched_keywords[category].append(keyword)

ranked = sorted(
    scores.items(),
    key=lambda item:item[1],
    reverse:True
    }
    best_category, best_score=ranked[0]
    second_score = ranked[1][1]
#Avoid guessing when there is too little evidence
#or two categories have similar evidence.
if best_score < 3 or best_score == second_score:
    return{
        "Issue_category":"Unknown",
        "confidence":"Low",
        "matched_keywords:"[],
        "scores":scores,
        "message":{
            "Unable to confidently classify this issue."
            "Review the screenshot or add relevent keywords:"
        },
    }
if best_score >=6 and best_score >= second_score + 3:
    confidence = "High"
else:
confidence = "Medium"
return{
    "Issue_category": best_category,
     "Confidence": confidence,
    "matched keywords": matched_keywords[best_category],
    "scores":scores,
    "message":f"Detected a possible(best_category).",
}
        
    
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
