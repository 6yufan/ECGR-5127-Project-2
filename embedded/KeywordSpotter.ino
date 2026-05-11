#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>

#include "model_data.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Keyword spotting model is larger than hello_world.
// Start with 80 KB. If AllocateTensors() fails, increase it.
constexpr int kTensorArenaSize = 1400 * 1024;
uint8_t* tensor_arena = nullptr;

const char* labels[] = {
  "silence",
  "unknown",
  "dog",
  "spottie"
};

const int num_labels = 4;

void printTensorInfo(TfLiteTensor* tensor, const char* name) {
  Serial.print(name);
  Serial.println(" tensor info:");

  Serial.print("  type: ");
  Serial.println(tensor->type);

  Serial.print("  dims: ");
  for (int i = 0; i < tensor->dims->size; i++) {
    Serial.print(tensor->dims->data[i]);
    if (i < tensor->dims->size - 1) {
      Serial.print(" x ");
    }
  }
  Serial.println();

  Serial.print("  scale: ");
  Serial.println(tensor->params.scale, 8);

  Serial.print("  zero_point: ");
  Serial.println(tensor->params.zero_point);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("Keyword Spotting Model Test");
  Serial.println("Initializing TensorFlow Lite Micro...");

  model = tflite::GetModel(keyword_model_int8_tflite);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version does not match TFLite Micro library.");
    Serial.print("Model version: ");
    Serial.println(model->version());
    Serial.print("Library schema version: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true);
  }

  static tflite::AllOpsResolver resolver;

  if (!psramFound()) {
    Serial.println("PSRAM not found. Please enable PSRAM in Tools menu.");
    while (true);
  }

  tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);

  if (tensor_arena == nullptr) {
    Serial.println("Failed to allocate tensor arena in PSRAM.");
    while (true);
  }

  Serial.print("Tensor arena allocated in PSRAM: ");
  Serial.print(kTensorArenaSize);
  Serial.println(" bytes");

  static tflite::MicroInterpreter static_interpreter(
    model,
    resolver,
    tensor_arena,
    kTensorArenaSize
  );

  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();

  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed.");
    Serial.println("Try increasing kTensorArenaSize.");
    while (true);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("TFLite model initialized successfully.");
  Serial.println();

  printTensorInfo(input, "Input");
  printTensorInfo(output, "Output");

  Serial.println();
  Serial.println("Filling input tensor with zeros for dummy inference...");

  int input_size = 1;
  for (int i = 0; i < input->dims->size; i++) {
    input_size *= input->dims->data[i];
  }

  Serial.print("Input size: ");
  Serial.println(input_size);

  if (input->type == kTfLiteInt8) {
    for (int i = 0; i < input_size; i++) {
      input->data.int8[i] = input->params.zero_point;
    }
  } else if (input->type == kTfLiteFloat32) {
    for (int i = 0; i < input_size; i++) {
      input->data.f[i] = 0.0f;
    }
  } else {
    Serial.println("Unsupported input tensor type.");
    while (true);
  }

  Serial.println("Running inference...");

  unsigned long start_time = micros();

  TfLiteStatus invoke_status = interpreter->Invoke();

  unsigned long end_time = micros();

  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed.");
    while (true);
  }

  Serial.println("Inference done.");

  Serial.print("Inference time: ");
  Serial.print(end_time - start_time);
  Serial.println(" us");

  Serial.println();
  Serial.println("Output scores:");

  int output_size = 1;
  for (int i = 0; i < output->dims->size; i++) {
    output_size *= output->dims->data[i];
  }

  for (int i = 0; i < output_size; i++) {
    float score;

    if (output->type == kTfLiteInt8) {
      int8_t q = output->data.int8[i];
      score = (q - output->params.zero_point) * output->params.scale;
    } else if (output->type == kTfLiteFloat32) {
      score = output->data.f[i];
    } else {
      Serial.println("Unsupported output tensor type.");
      return;
    }

    if (i < num_labels) {
      Serial.print(labels[i]);
    } else {
      Serial.print("class ");
      Serial.print(i);
    }

    Serial.print(": ");
    Serial.println(score, 6);
  }
}

void loop() {
}