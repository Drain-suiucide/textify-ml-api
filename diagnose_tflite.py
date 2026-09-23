import tensorflow as tf


models = [
    "models/mobilenet_v2_fp32.tflite",
    "models/mobilenet_v2_int8.tflite",
]


for model_path in models:
    print("\n" + "=" * 60)
    print(model_path)
    print("=" * 60)

    interpreter = tf.lite.Interpreter(
        model_path=model_path
    )

    interpreter.allocate_tensors()

    print("\nInput:")
    print(interpreter.get_input_details())

    print("\nOutput:")
    print(interpreter.get_output_details())

    print("\nSignature:")
    print(interpreter.get_signature_list())