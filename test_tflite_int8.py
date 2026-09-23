import time

import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = "models/mobilenet_v2_int8.tflite"
IMAGE_PATH = "извинись.jpg"


print("Loading INT8 TFLite model...")

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


# Get INT8 quantization parameters
input_scale, input_zero_point = input_details[0]["quantization"]

output_scale, output_zero_point = output_details[0]["quantization"]

print("\nInput quantization:")
print(f"scale={input_scale}")
print(f"zero_point={input_zero_point}")

print("\nOutput quantization:")
print(f"scale={output_scale}")
print(f"zero_point={output_zero_point}")


# Convert FP32 input to INT8
input_data = (
    input_data / input_scale + input_zero_point
)

input_data = np.clip(
    input_data,
    -128,
    127,
).astype(np.int8)


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


# Dequantize output
predictions = (
    output_data.astype(np.float32) - output_zero_point
) * output_scale


predictions = predictions[0]

top_indices = np.argsort(predictions)[-3:][::-1]


print("\nTop 3 predictions:")

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