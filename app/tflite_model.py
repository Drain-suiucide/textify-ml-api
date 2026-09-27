import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = "models/mobilenet_v2_fp32.tflite"
CLASS_INDEX_PATH = Path("models/imagenet_class_index.json")


class TFLiteModel:
    def __init__(self, model_path: str = MODEL_PATH):
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        with CLASS_INDEX_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        self.class_index = {
            int(index): value
            for index, value in data.items()
        }

    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        image = image.resize((224, 224))

        image_array = np.array(
            image,
            dtype=np.float32,
        )
        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        image_array = (
            tf.keras.applications.mobilenet_v2
            .preprocess_input(image_array)
        )

        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            image_array,
        )

        self.interpreter.invoke()

        output = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )

        top_indices = np.argsort(
            output[0]
        )[-3:][::-1]

        predictions = []

        for class_index in top_indices:
            _, label = self.class_index[
                int(class_index)
            ]

            confidence = float(
                output[0][class_index]
            )

            predictions.append(
                {
                    "label": label,
                    "confidence": round(
                        confidence,
                        4,
                    ),
                }
            )

        return {
            "model": "MobileNetV2 TFLite FP32",
            "predictions": predictions,
        }


model = TFLiteModel()
