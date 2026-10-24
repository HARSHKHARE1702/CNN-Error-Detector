
# ai_engine.py

import re

import easyocr
import numpy as np
import tensorflow as tf
from PIL import Image


# ---------------- CONFIGURATION ----------------

MODEL_PATH = "error_cnn.keras"
IMAGE_SIZE = (128, 128)
ERROR_THRESHOLD = 0.50


# ---------------- LOAD MODELS ONCE ----------------

print("Loading CNN model...")
cnn_model = tf.keras.models.load_model(MODEL_PATH)

print("Loading OCR reader...")
ocr_reader = easyocr.Reader(["en"], gpu=False)

print("AI engine is ready.")


# ---------------- OCR PATTERNS ----------------

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


def matches_any(text, patterns):
    return any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in patterns
    )


# ---------------- ISSUE KEYWORDS ----------------

ISSUE_KEYWORDS = {
    "Login": {
        "login": 3,
        "log in": 3,
        "sign in": 3,
        "signin": 3,
        "authentication": 3,
        "invalid credentials": 5,
        "incorrect password": 5,
        "wrong password": 5,
        "password mismatch": 4,
        "unauthorized": 3,
        "session expired": 3,
        "login failed": 5,
        "invalid username": 4,
    },
    "Registration": {
        "register": 3,
        "registration": 4,
        "signup": 3,
        "sign up": 3,
        "create account": 4,
        "already exists": 3,
        "user already exists": 5,
        "duplicate user": 5,
        "registration failed": 5,
        "otp": 2,
        "verification code": 3,
        "email already exists": 5,
        "mobile number already registered": 5,
    },
    "MPIN": {
        "mpin": 5,
        "m-pin": 5,
        "m pin": 5,
        "incorrect pin": 4,
        "invalid pin": 4,
        "wrong pin": 4,
        "pin verification": 4,
        "pin validation": 4,
        "pin mismatch": 4,
        "reset pin": 4,
        "set pin": 3,
        "forgot pin": 4,
        "mpin failed": 5,
        "mpin verification": 5,
        "transaction pin": 2,
    },
}


# ---------------- CNN PREDICTION ----------------

def detect_error(image: Image.Image):
    resized = image.convert("RGB").resize(IMAGE_SIZE)

    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Model output is probability of Normal.
    probability_normal = float(
        cnn_model.predict(image_array, verbose=0).reshape(-1)[0]
    )

    probability_error = 1.0 - probability_normal

    prediction = (
        "Error"
        if probability_error >= ERROR_THRESHOLD
        else "Normal"
    )

    return prediction, probability_error, probability_normal


# ---------------- OCR EXTRACTION ----------------

def extract_error_message(image: Image.Image):
    results = ocr_reader.readtext(
        np.asarray(image.convert("RGB")),
        detail=1,
        paragraph=False,
    )

    detected_lines = []

    for bbox, text, ocr_confidence in results:
        text = text.strip()

        if not text:
            continue

        top = min(point[1] for point in bbox)
        left = min(point[0] for point in bbox)

        detected_lines.append((top, left, text))

    detected_lines.sort(key=lambda item: (item[0], item[1]))
    all_lines = [item[2] for item in detected_lines]

    selected = set()

    for index, line in enumerate(all_lines):
        is_error = matches_any(line, ERROR_PATTERNS)
        is_trace = matches_any(line, TRACE_PATTERNS)

        if is_error or is_trace:
            selected.add(index)

            # Include adjacent lines for wrapped messages.
            for nearby in range(
                max(0, index - 1),
                min(len(all_lines), index + 2),
            ):
                selected.add(nearby)

    # Keep traceback lines through the likely final exception.
    for index, line in enumerate(all_lines):
        if re.search(r"\bTraceback\b", line, re.IGNORECASE):
            for following in range(index, len(all_lines)):
                selected.add(following)

                if (
                    following > index
                    and matches_any(all_lines[following], ERROR_PATTERNS)
                    and not matches_any(
                        all_lines[following], TRACE_PATTERNS
                    )
                ):
                    break

    extracted_lines = [all_lines[i] for i in sorted(selected)]

    # Remove duplicates without changing order.
    unique_lines = []
    seen = set()

    for line in extracted_lines:
        key = line.casefold()

        if key not in seen:
            seen.add(key)
            unique_lines.append(line)

    return "\n".join(unique_lines).strip(), all_lines


# ---------------- ISSUE CLASSIFICATION ----------------

def classify_issue(error_text: str, all_ocr_text: str = ""):
    combined_text = f"{error_text}\n{all_ocr_text}".casefold()

    scores = {}
    matched = {}

    for category, keywords in ISSUE_KEYWORDS.items():
        scores[category] = 0
        matched[category] = []

        for keyword, weight in keywords.items():
            if keyword in combined_text:
                scores[category] += weight
                matched[category].append(keyword)

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_category, best_score = ranked[0]
    second_score = ranked[1][1]

    # Avoid a forced classification without enough evidence.
    if best_score < 3 or best_score == second_score:
        return {
            "category": "Unknown",
            "confidence": "Low",
            "matched_keywords": [],
            "scores": scores,
        }

    if best_score >= 6 and best_score >= second_score + 3:
        confidence = "High"
    else:
        confidence = "Medium"

    return {
        "category": best_category,
        "confidence": confidence,
        "matched_keywords": matched[best_category],
        "scores": scores,
    }


# ---------------- PUBLIC ANALYSIS FUNCTION ----------------

def analyze_image(image: Image.Image):
    prediction, error_prob, normal_prob = detect_error(image)

    result = {
        "is_error": prediction == "Error",
        "prediction": prediction,
        "probability_error": round(error_prob, 4),
        "probability_normal": round(normal_prob, 4),
        "category": "NotApplicable",
        "category_confidence": "NotApplicable",
        "extracted_error": "",
        "text_found": False,
        "matched_keywords": [],
    }

    # Only run OCR and classification if CNN predicts Error.
    if prediction == "Error":
        extracted_text, all_text = extract_error_message(image)

        classification = classify_issue(
            error_text=extracted_text,
            all_ocr_text="\n".join(all_text),
        )

        result.update({
            "category": classification["category"],
            "category_confidence": classification["confidence"],
            "extracted_error": extracted_text,
            "text_found": bool(extracted_text),
            "matched_keywords": classification["matched_keywords"],
        })

    return result
