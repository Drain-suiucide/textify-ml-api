# AI Image Classifier API

Production-oriented computer vision inference API built with **FastAPI, TensorFlow Lite, Docker and GitHub Actions**.

The project exposes a REST API for image classification using a **MobileNetV2** model converted to TensorFlow Lite. It includes input validation, automated API tests, model optimization experiments, latency benchmarking, containerization, health checks and CI/CD automation.

## Overview

The service accepts an image through a REST endpoint and returns the top-3 ImageNet predictions with confidence scores.

The inference pipeline is designed to avoid external network dependencies during prediction. ImageNet class labels are bundled with the application and loaded locally.

### Main capabilities

* FastAPI REST API
* TensorFlow / TensorFlow Lite inference
* MobileNetV2 ImageNet classifier
* FP32 TFLite model
* INT8 quantization experiment
* Input validation and file-size limits
* Pydantic response schemas
* Automated API tests with pytest
* Docker containerization
* Docker Compose
* Container health check
* Non-root application user
* GitHub Actions CI
* Automated Docker image build
* Local offline class-label lookup

## Architecture

```text
Client
  │
  │ multipart/form-data
  ▼
FastAPI
  │
  ├── Request validation
  ├── File type validation
  ├── File size validation
  └── Image integrity validation
  │
  ▼
TFLiteModel
  │
  ├── RGB conversion
  ├── Resize → 224 × 224
  ├── MobileNetV2 preprocessing
  └── TFLite inference
  │
  ▼
Top-3 predictions
  │
  ▼
Pydantic response
```

Containerized deployment:

```text
Windows
   │
   ▼
WSL 2
   │
   ▼
Docker Engine
   │
   ▼
Docker Compose
   │
   ▼
FastAPI container
   │
   ▼
TensorFlow Lite MobileNetV2
```

## Project Structure

```text
textify-ml-api/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── tflite_model.py
│   ├── schemas.py
│   ├── convert_model.py
│   └── quantize_model.py
│
├── models/
│   ├── mobilenet_v2_fp32.tflite
│   ├── mobilenet_v2_int8.tflite
│   └── imagenet_class_index.json
│
├── tests/
│   └── test_api.py
│
├── benchmark_fp32.py
├── benchmark_tflite.py
├── benchmark_tflite_int8.py
├── diagnose_tflite.py
├── inspect_model.py
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitattributes
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

## API

### Health check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model": "MobileNetV2"
}
```

### Image classification

```http
POST /predict
Content-Type: multipart/form-data
```

Form field:

```text
file
```

Example with cURL:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -F "file=@image.jpg;type=image/jpeg"
```

Example response:

```json
{
  "status": "success",
  "filename": "image.jpg",
  "model": "MobileNetV2 TFLite FP32",
  "predictions": [
    {
      "label": "minivan",
      "confidence": 0.5283
    },
    {
      "label": "limousine",
      "confidence": 0.1258
    },
    {
      "label": "lakeside",
      "confidence": 0.0243
    }
  ]
}
```

## Validation

The API validates uploaded files before inference.

Supported formats:

```text
JPEG
PNG
WEBP
```

Maximum file size:

```text
5 MB
```

The API returns appropriate HTTP errors for:

* unsupported content types
* unsupported image formats
* empty files
* corrupted images
* files exceeding the size limit
* inference failures

## Model

The primary model is **MobileNetV2 pretrained on ImageNet**.

Input:

```text
224 × 224 × 3
```

Output:

```text
1000 ImageNet classes
```

The production API uses the **TensorFlow Lite FP32 model**.

The ImageNet class mapping is stored locally in:

```text
models/imagenet_class_index.json
```

This prevents inference from depending on an external network request.

## Model Optimization

An INT8 quantized version of MobileNetV2 was also created and benchmarked.

### Model size

| Model       |     Size |
| ----------- | -------: |
| TFLite FP32 | 13.35 MB |
| TFLite INT8 |  3.81 MB |

INT8 quantization reduced the model size by approximately **71.5%**.

### Local benchmark

Benchmark configuration:

```text
Runs: 50
Hardware: Intel Core i3-10100
Runtime: TensorFlow Lite
```

| Model |    Average |     Median |    Minimum |    Maximum |
| ----- | ---------: | ---------: | ---------: | ---------: |
| FP32  |   12.86 ms |   12.74 ms |   11.73 ms |   15.67 ms |
| INT8  | 1756.72 ms | 1754.90 ms | 1746.67 ms | 1798.02 ms |

### Engineering finding

INT8 significantly reduced model size, but it **did not improve latency in this particular runtime environment**.

The FP32 TFLite model therefore remains the production model used by the API.

This demonstrates an important optimization principle: **quantization can reduce memory/storage requirements without necessarily improving inference latency on every hardware/runtime combination**.

## Testing

The project contains automated API tests covering:

```text
7 tests
```

Test coverage includes:

```text
/root
/health
valid image inference
invalid content type
empty file
corrupted image
file size limit
```

Run locally:

```bash
pytest -v
```

Expected result:

```text
7 passed
```

## Docker

The application is containerized using Docker.

Build:

```bash
docker build -t textify-ml-api:latest .
```

Run:

```bash
docker run --rm -p 8000:8000 textify-ml-api:latest
```

Or use Docker Compose:

```bash
docker compose up --build -d
```

Check status:

```bash
docker compose ps
```

The container exposes:

```text
http://127.0.0.1:8000
```

### Container health check

The image includes a Docker `HEALTHCHECK` based on the `/health` endpoint.

Example:

```text
textify-ml-api   Up (...) (healthy)
```

The container also runs the application as a dedicated non-root user.

## CI/CD

GitHub Actions automatically performs:

```text
Push / Pull Request
        │
        ▼
Run pytest
        │
        ▼
Build Docker image
```

The Docker build depends on successful tests.

Workflow:

```text
.github/workflows/ci.yml
```

This ensures that changes are automatically validated before being considered ready.

## Local Development

### Windows + WSL 2

The project was developed on Windows with Ubuntu 22.04 running under WSL 2.

Project location:

```text
F:\textify-ml-api
```

Inside WSL:

```text
/mnt/f/textify-ml-api
```

Docker Engine runs inside the WSL 2 Ubuntu environment.

### Python environment

The application uses:

```text
Python 3.10
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app.main:app --reload
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Reproducibility

The repository contains:

* pinned Python dependencies
* TFLite model files
* ImageNet class mapping
* Docker configuration
* automated tests
* CI configuration

The Docker image packages the application and model artifacts into a reproducible runtime environment.

## Engineering Considerations

### External dependency during inference

The original implementation used TensorFlow's built-in class decoding functionality, which could attempt to download the ImageNet class mapping.

This was identified during container testing because inference stalled when the container attempted to access Google Cloud.

The implementation was changed to load:

```text
models/imagenet_class_index.json
```

locally.

Inference therefore no longer requires a third-party HTTP request.

### Runtime-specific optimization

The INT8 model is substantially smaller than FP32, but the local benchmark showed significantly higher latency.

Instead of assuming that quantization automatically improves performance, both models were benchmarked under the same environment and the FP32 model was selected for the API based on the observed result.

## Future Improvements

Possible extensions include:

* asynchronous/background inference for batch workloads
* structured application logging
* Prometheus metrics
* model versioning
* batch inference endpoint
* image preprocessing pipeline abstraction
* model warm-up
* inference latency metrics
* cloud deployment
* mobile/on-device deployment using TFLite
* automated image/model artifact versioning

## License

This project is intended as a portfolio and engineering demonstration project.

