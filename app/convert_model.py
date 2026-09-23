from pathlib import Path

import tensorflow as tf

from app.model import model


OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "mobilenet_v2_fp32.tflite"


print("Starting TensorFlow Lite conversion...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

OUTPUT_PATH.write_bytes(tflite_model)

print(f"Model saved to: {OUTPUT_PATH}")
print(f"Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")