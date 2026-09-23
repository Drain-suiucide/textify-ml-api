import time

import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = "models/mobilenet_v2_fp32.tflite"
IMAGE_PATH = "извинись.jpg"


print("Loading TFLite model...")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput:")
print(input_details)

print("\nOutput:")
print(output_details)


image = Image.open(IMAGE_PATH).convert("RGB")
image = image.resize((224, 224))

input_data = np.array(image, dtype=np.float32)
input_data = np.expand_dims(input_data, axis=0)

input_data = tf.keras.applications.mobilenet_v2.preprocess_input(
    input_data
)


interpreter.set_tensor(
    input_details[0]["index"],
    input_data,
)


start = time.perf_counter()

interpreter.invoke()

end = time.perf_counter()


output_data = interpreter.get_tensor(
    output_details[0]["index"]
)

predictions = output_data[0]

top_indices = np.argsort(predictions)[-3:][::-1]

print("\nTop 3 predictions:")

for index in top_indices:
    print(
        f"class_id={index}, "
        f"confidence={predictions[index]:.4f}"
    )

print(f"\nInference time: {(end - start) * 1000:.2f} ms")