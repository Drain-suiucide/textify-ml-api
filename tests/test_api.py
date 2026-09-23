from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def create_test_image() -> bytes:
    image = Image.new(
        "RGB",
        (224, 224),
        color="white",
    )

    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    return buffer.getvalue()


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "MobileNetV2"


def test_predict_valid_image():
    image = create_test_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.jpg",
                image,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert data["filename"] == "test.jpg"

    assert data["model"] == "MobileNetV2 TFLite FP32"

    assert len(data["predictions"]) == 3

    for prediction in data["predictions"]:
        assert "label" in prediction
        assert "confidence" in prediction

        assert 0.0 <= prediction["confidence"] <= 1.0


def test_predict_invalid_content_type():
    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                b"hello world",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415


def test_predict_empty_file():
    response = client.post(
        "/predict",
        files={
            "file": (
                "empty.jpg",
                b"",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400


def test_predict_corrupted_image():
    response = client.post(
        "/predict",
        files={
            "file": (
                "corrupted.jpg",
                b"this is not an image",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400


def test_predict_file_too_large():
    large_file = b"0" * (5 * 1024 * 1024 + 1)

    response = client.post(
        "/predict",
        files={
            "file": (
                "large.jpg",
                large_file,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 413