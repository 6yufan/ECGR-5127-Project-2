input_file = "../final_model/keyword_model_int8.tflite"
output_file = "model_data.cc"

with open(input_file, "rb") as f:
    data = f.read()

with open(output_file, "w") as f:
    f.write('#include "model_data.h"\n\n')
    f.write("const unsigned char keyword_model_int8_tflite[] = {\n")

    for i, b in enumerate(data):
        if i % 12 == 0:
            f.write("  ")
        f.write(f"0x{b:02x}, ")
        if i % 12 == 11:
            f.write("\n")

    f.write("\n};\n\n")
    f.write(f"const unsigned int keyword_model_int8_tflite_len = {len(data)};\n")

print("Created model_data.cc")
print("Model size:", len(data), "bytes")