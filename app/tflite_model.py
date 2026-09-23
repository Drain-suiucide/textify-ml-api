import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = "models/mobilenet_v2_fp32.tflite"


class TFLiteModel:
    def __init__(self, model_path: str = MODEL_PATH):
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path
        )

        self.interpreter.allocate_tensors()

        self.input_details = (
            self.interpreter.get_input_details()
        )

        self.output_details = (
            self.interpreter.get_output_details()
        )

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

        decoded = (
            tf.keras.applications.mobilenet_v2
            .decode_predictions(
                output,
                top=3,
            )[0]
        )

        predictions = []

        for _, label, confidence in decoded:
            predictions.append(
                {
                    "label": label,
                    "confidence": round(
                        float(confidence),
                        4,
                    ),
                }
            )

        return {
            "model": "MobileNetV2 TFLite FP32",
            "predictions": predictions,
        }


model = TFLiteModel()