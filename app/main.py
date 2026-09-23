from io import BytesIO
from app.schemas import PredictionResponse
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.tflite_model import model


app = FastAPI(
    title="AI Image Classifier API",
    description="Computer vision inference API powered by MobileNetV2.",
    version="0.3.0",
)

MAX_FILE_SIZE = 5 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


@app.get("/")
def root():
    return {
        "message": "AI Image Classifier API is running",
        "status": "ok",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "MobileNetV2",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image type. Use JPEG, PNG, or WEBP.",
        )

    contents = await file.read(MAX_FILE_SIZE + 1)

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Maximum size is 5 MB.",
        )

    try:
        with Image.open(BytesIO(contents)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=415,
                    detail="Unsupported image format.",
                )

            image.verify()

        image = Image.open(BytesIO(contents))

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    except OSError:
        raise HTTPException(
            status_code=400,
            detail="The image file is corrupted or unreadable.",
        )

    try:
        result = model.predict(image)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(exc)}",
        )

    return {
        "status": "success",
        "filename": file.filename,
        "model": result["model"],
        "predictions": result["predictions"],
    }