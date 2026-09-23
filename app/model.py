
import numpy as np
import tensorflow as tf

from PIL import Image


# MobileNetV2 pretrained on ImageNet
model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=True,
)

preprocess_input = (
    tf.keras.applications.mobilenet_v2.preprocess_input
)


def predict_image(image: Image.Image) -> dict:
    """
    Classify an image using pretrained MobileNetV2.
    """

    # Convert to RGB and resize to model input dimensions
    image = image.convert("RGB")
    image = image.resize((224, 224))

    # Convert image to a NumPy array
    image_array = np.array(image, dtype=np.float32)

    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)

    # Apply MobileNetV2 preprocessing
    image_array = preprocess_input(image_array)

    # Run inference
    predictions = model.predict(image_array, verbose=0)

    # Decode top 3 predictions
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
        predictions,
        top=3,
    )[0]

    results = []

    for _, label, confidence in decoded:
        results.append({
            "label": label,
            "confidence": round(float(confidence), 4),
        })

    return {
        "model": "MobileNetV2",
        "predictions": results,
    }