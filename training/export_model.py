import tensorflow as tf
import numpy as np

H5_MODEL = "../final_model/keyword_model.h5"
TFLITE_MODEL = "../final_model/keyword_model_int8.tflite"

model = tf.keras.models.load_model(H5_MODEL)

# 查看模型输入 shape
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

# 你的 Arduino 输出显示 input shape 是 1 x 124 x 129 x 1
INPUT_SHAPE = (1, 124, 129, 1)


def representative_dataset():
    for _ in range(100):
        # 注意：这里最好换成真实训练数据的 spectrogram / MFCC
        sample = np.random.rand(*INPUT_SHAPE).astype(np.float32)
        yield [sample]


converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 启用量化
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 提供 representative dataset，帮助确定 int8 scale / zero_point
converter.representative_dataset = representative_dataset

# 强制只使用 int8 TFLite built-in ops
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

# 强制输入输出也是 int8
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open(TFLITE_MODEL, "wb") as f:
    f.write(tflite_model)

print("Saved:", TFLITE_MODEL)


# 检查转换结果
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput details:")
print(input_details)

print("\nOutput details:")
print(output_details)

print("\nInput dtype:", input_details[0]["dtype"])
print("Output dtype:", output_details[0]["dtype"])