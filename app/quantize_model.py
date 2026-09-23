from pathlib import Path
from app.model import model
import numpy as np
import tensorflow as tf
from PIL import Image


OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

CALIBRATION_DIR = Path("calibration_images")

OUTPUT_PATH = OUTPUT_DIR / "mobilenet_v2_int8.tflite"


def representative_dataset():
    image_paths = list(CALIBRATION_DIR.glob("*.jpg"))
    image_paths += list(CALIBRATION_DIR.glob("*.jpeg"))
    image_paths += list(CALIBRATION_DIR.glob("*.png"))

    if not image_paths:
        raise RuntimeError(
            "No calibration images found in calibration_images/"
        )

    for image_path in image_paths[:100]:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))

        image_array = np.array(
            image,
            dtype=np.float32,
        )

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array
        )

        yield [image_array]


print("Starting INT8 quantization...")
print("Using real calibration images...")


converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8


tflite_model = converter.convert()

OUTPUT_PATH.write_bytes(tflite_model)


print(f"Model saved to: {OUTPUT_PATH}")
print(
    f"Model size: "
    f"{len(tflite_model) / 1024 / 1024:.2f} MB"
)