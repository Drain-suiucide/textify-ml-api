import time

import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = "models/mobilenet_v2_fp32.tflite"
IMAGE_PATH = "извинись.jpg"


interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


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


# Warm-up
for _ in range(5):
    interpreter.invoke()


# Benchmark
runs = 50
times = []

for _ in range(runs):
    start = time.perf_counter()

    interpreter.invoke()

    end = time.perf_counter()

    times.append((end - start) * 1000)


output_data = interpreter.get_tensor(
    output_details[0]["index"]
)

predictions = output_data[0]

top_indices = np.argsort(predictions)[-3:][::-1]


print("Top 3 predictions:")

for index in top_indices:
    print(
        f"class_id={index}, "
        f"confidence={predictions[index]:.4f}"
    )


print("\nBenchmark:")
print(f"Runs: {runs}")
print(f"Average inference time: {np.mean(times):.2f} ms")
print(f"Median inference time: {np.median(times):.2f} ms")
print(f"Min inference time: {np.min(times):.2f} ms")
print(f"Max inference time: {np.max(times):.2f} ms")