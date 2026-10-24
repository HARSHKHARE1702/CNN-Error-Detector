
# api.py

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from ai_engine import analyze_image


app = FastAPI(
    title="Screenshot AI Support Bot",
    description="CNN-based error detection with OCR and issue classification",
    version="1.0.0",
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Button definitions returned to the client.
BUTTONS = {
    "Login": [
        {
            "id": "login_help",
            "label": "Login Help",
            "type": "primary",
        }
    ],
    "Registration": [
        {
            "id": "registration_help",
            "label": "Registration Help",
            "type": "primary",
        }
    ],
    "MPIN": [
        {
            "id": "mpin_help",
            "label": "MPIN Help",
            "type": "primary",
        }
    ],
    "Unknown": [
        {
            "id": "select_issue",
            "label": "Select Your Issue",
            "type": "secondary",
        }
    ],
}


@app.get("/")
def home():
    return {
        "service": "Screenshot AI Support Bot",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "ai_engine": "loaded",
    }


@app.post("/analyze-screenshot")
async def analyze_screenshot(
    file: UploadFile = File(...),
):
    # Validate declared content type.
    if file.content_type not in {
        "image/png",
        "image/jpeg",
        "image/webp",
    }:
        raise HTTPException(
            status_code=415,
            detail="Upload a PNG, JPEG, or WebP screenshot.",
        )

    # Read at most 10 MB + 1 byte to enforce the limit.
    contents = await file.read(MAX_FILE_SIZE + 1)

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Screenshot exceeds the 10 MB limit.",
        )

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    try:
        analysis = analyze_image(image)
    except Exception:
        # Do not expose internal paths or model details to clients.
        raise HTTPException(
            status_code=500,
            detail="Screenshot analysis failed.",
        )

    category = analysis["category"]

    # Unknown, unreadable, or uncertain cases offer manual selection.
    if category not in {"Login", "Registration", "MPIN"}:
        category = "Unknown"

    return {
        "success": True,
        "prediction": {
            "is_error": analysis["is_error"],
            "category": category,
            "confidence": analysis["category_confidence"],
            "probability_error": analysis["probability_error"],
        },
        "extracted_error": analysis["extracted_error"],
        "text_found": analysis["text_found"],
        "ui": {
            "message": (
                f"We detected a possible {category} issue. "
                "How can we help?"
                if category != "Unknown" and analysis["is_error"]
                else "Please select the type of issue you are experiencing."
            ),
            "buttons": (
                BUTTONS[category]
                if analysis["is_error"]
                else []
            ),
        },
    }
