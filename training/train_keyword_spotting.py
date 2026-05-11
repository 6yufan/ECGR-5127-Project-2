import os
import pathlib
import csv

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

SAMPLE_RATE = 16000
CLIP_DURATION = 1
SAMPLES_PER_CLIP = SAMPLE_RATE * CLIP_DURATION

DATA_DIR = pathlib.Path("data")
CUSTOM_WORD = "blueberry"

COMMANDS = ["backward", CUSTOM_WORD]
LABELS = ["silence", "unknown"] + COMMANDS

BATCH_SIZE = 32
EPOCHS = 25

TEMPERATURE = 3.0
ALPHA = 0.5


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1)

    audio_len = tf.shape(audio)[0]

    audio = tf.cond(
        audio_len < SAMPLES_PER_CLIP,
        lambda: tf.pad(audio, [[0, SAMPLES_PER_CLIP - audio_len]]),
        lambda: audio[:SAMPLES_PER_CLIP]
    )

    return audio


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    folder_name = parts[-2]

    label = tf.cond(
        tf.reduce_any(folder_name == COMMANDS),
        lambda: folder_name,
        lambda: tf.constant("unknown")
    )

    return label


def get_waveform_and_label(file_path):
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    label = get_label(file_path)
    return waveform, label


def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(
        waveform,
        frame_length=256,
        frame_step=128
    )

    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]

    return spectrogram


def label_to_id(label):
    return tf.argmax(label == LABELS)


def preprocess(file_path):
    waveform, label = get_waveform_and_label(file_path)
    spec = get_spectrogram(waveform)
    label_id = label_to_id(label)
    return spec, label_id


def build_teacher_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes)
    ])

    return model


def build_student_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv2D(8, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes)
    ])

    return model


class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher

        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    def compile(
        self,
        optimizer,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.5,
        temperature=3.0,
    ):
        super().compile(optimizer=optimizer)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    @property
    def metrics(self):
        return [
            self.train_accuracy,
            self.val_accuracy,
        ]

    def train_step(self, data):
        x, y = data

        teacher_logits = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_logits = self.student(x, training=True)

            student_loss = self.student_loss_fn(y, student_logits)

            teacher_soft = tf.nn.softmax(
                teacher_logits / self.temperature,
                axis=1
            )

            student_soft = tf.nn.softmax(
                student_logits / self.temperature,
                axis=1
            )

            distillation_loss = self.distillation_loss_fn(
                teacher_soft,
                student_soft
            ) * (self.temperature ** 2)

            loss = self.alpha * student_loss + (1.0 - self.alpha) * distillation_loss

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_accuracy.update_state(y, student_logits)

        return {
            "accuracy": self.train_accuracy.result(),
            "loss": loss,
            "student_loss": student_loss,
            "distillation_loss": distillation_loss,
        }

    def test_step(self, data):
        x, y = data

        student_logits = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, student_logits)

        self.val_accuracy.update_state(y, student_logits)

        return {
            "accuracy": self.val_accuracy.result(),
            "loss": student_loss,
            "student_loss": student_loss,
        }

    def call(self, inputs):
        return self.student(inputs)


def get_predictions(model, dataset):
    y_true = []
    y_pred = []

    for specs, labels in dataset:
        logits = model.predict(specs, verbose=0)
        preds = tf.argmax(logits, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

    return np.array(y_true), np.array(y_pred)


def estimate_macs(model):
    total_macs = 0

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            output_shape = layer.output.shape

            out_h = int(output_shape[1])
            out_w = int(output_shape[2])
            out_c = int(output_shape[3])

            kernel_h, kernel_w = layer.kernel_size
            in_c = int(layer.input.shape[-1])

            layer_macs = out_h * out_w * out_c * kernel_h * kernel_w * in_c
            total_macs += layer_macs

        elif isinstance(layer, tf.keras.layers.Dense):
            input_units = int(layer.input.shape[-1])
            output_units = int(layer.output.shape[-1])

            layer_macs = input_units * output_units
            total_macs += layer_macs

    return total_macs


def plot_training_curves(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    history_dict = history.history

    print("\nAvailable history keys:")
    print(list(history_dict.keys()))

    loss = None
    val_loss = None

    possible_loss_keys = ["loss", "student_loss"]
    possible_val_loss_keys = ["val_loss", "val_student_loss"]

    for key in possible_loss_keys:
        if key in history_dict:
            loss = history_dict[key]
            break

    for key in possible_val_loss_keys:
        if key in history_dict:
            val_loss = history_dict[key]
            break

    acc = None
    val_acc = None

    possible_acc_keys = [
        "accuracy",
        "sparse_categorical_accuracy"
    ]

    possible_val_acc_keys = [
        "val_accuracy",
        "val_sparse_categorical_accuracy"
    ]

    for key in possible_acc_keys:
        if key in history_dict:
            acc = history_dict[key]
            break

    for key in possible_val_acc_keys:
        if key in history_dict:
            val_acc = history_dict[key]
            break

    if loss is not None and val_loss is not None:
        epochs_range = range(1, len(loss) + 1)

        plt.figure()
        plt.plot(epochs_range, loss, label="Train Loss")
        plt.plot(epochs_range, val_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Student Training and Validation Loss")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_dir, "loss_curve.png"))
        plt.close()

        print("Saved:", os.path.join(save_dir, "loss_curve.png"))
    else:
        print("Warning: loss or validation loss was not found.")

    if acc is not None and val_acc is not None:
        epochs_range = range(1, len(acc) + 1)

        plt.figure()
        plt.plot(epochs_range, acc, label="Train Accuracy")
        plt.plot(epochs_range, val_acc, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Student Training and Validation Accuracy")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
        plt.close()

        print("Saved:", os.path.join(save_dir, "accuracy_curve.png"))

        plt.figure()
        plt.plot(epochs_range, acc, label="Train Accuracy")
        plt.plot(epochs_range, val_acc, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Train vs Validation Accuracy")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_dir, "train_val_accuracy.png"))
        plt.close()

        print("Saved:", os.path.join(save_dir, "train_val_accuracy.png"))
    else:
        print("Warning: accuracy or validation accuracy was not found.")
        print("No accuracy plot was generated.")


def plot_confusion_matrix(cm, labels, save_path):
    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="gray",
        cbar=True
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_confusion_matrix_csv(cm, labels, save_path):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["True/Predicted"] + labels)

        for i, row in enumerate(cm):
            writer.writerow([labels[i]] + list(row))


def calculate_frr(y_true, y_pred, class_id):
    class_indices = np.where(y_true == class_id)[0]

    if len(class_indices) == 0:
        return None

    false_rejections = np.sum(y_pred[class_indices] != class_id)
    total_target_samples = len(class_indices)

    frr = false_rejections / total_target_samples

    return frr


def convert_to_full_int8_tflite(model, train_ds, tflite_path):
    def representative_dataset():
        for specs, labels in train_ds.take(100):
            sample = specs[0:1]
            sample = tf.cast(sample, tf.float32)
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = representative_dataset

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print("Saved full int8 TFLite model:", tflite_path)

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\nTFLite input details:")
    print(input_details)

    print("\nTFLite output details:")
    print(output_details)

    print("\nInput dtype:", input_details[0]["dtype"])
    print("Output dtype:", output_details[0]["dtype"])

    if input_details[0]["dtype"] == np.int8 and output_details[0]["dtype"] == np.int8:
        print("Full int8 conversion successful.")
    else:
        print("Warning: model is not full int8.")

    return input_details, output_details

def stratified_split(data_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)

    train_files = []
    val_files = []
    test_files = []

    class_folders = [p for p in pathlib.Path(data_dir).iterdir() if p.is_dir()]

    for class_folder in class_folders:
        class_files = list(class_folder.glob("*.wav"))

        if len(class_files) == 0:
            continue

        class_files = np.array([str(p) for p in class_files])
        rng.shuffle(class_files)

        n = len(class_files)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        train_files.extend(class_files[:n_train])
        val_files.extend(class_files[n_train:n_train + n_val])
        test_files.extend(class_files[n_train + n_val:])

        print(
            f"{class_folder.name}: total={n}, "
            f"train={n_train}, val={n_val}, test={n - n_train - n_val}"
        )

    rng.shuffle(train_files)
    rng.shuffle(val_files)
    rng.shuffle(test_files)

    return train_files, val_files, test_files

def main():
    os.makedirs("./images", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    os.makedirs("../final_model", exist_ok=True)

    files = tf.io.gfile.glob(str(DATA_DIR / "*" / "*.wav"))

    if len(files) == 0:
        raise RuntimeError("No wav files found. Please check DATA_DIR.")

    files = tf.random.shuffle(files)

    n = len(files)
    train_files, val_files, test_files = stratified_split(
        DATA_DIR,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=SEED
    )

    print("Labels:", LABELS)
    print("Total files:", n)

    print("Total files:", len(train_files) + len(val_files) + len(test_files))
    print("Train files:", len(train_files))
    print("Validation files:", len(val_files))
    print("Test files:", len(test_files))

    train_ds = tf.data.Dataset.from_tensor_slices(train_files)
    val_ds = tf.data.Dataset.from_tensor_slices(val_files)
    test_ds = tf.data.Dataset.from_tensor_slices(test_files)

    train_ds = train_ds.map(preprocess).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

    for spec, label in train_ds.take(1):
        input_shape = spec.shape[1:]
        print("Input shape:", input_shape)

    num_classes = len(LABELS)

    # ============================================================
    # 1. Train teacher model
    # ============================================================
    teacher = build_teacher_model(input_shape, num_classes)

    teacher.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    print("\n===== Teacher Model Summary =====")
    teacher.summary()

    teacher_history = teacher.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    teacher_train_loss, teacher_train_acc = teacher.evaluate(train_ds, verbose=0)
    teacher_val_loss, teacher_val_acc = teacher.evaluate(val_ds, verbose=0)
    teacher_test_loss, teacher_test_acc = teacher.evaluate(test_ds, verbose=0)

    print("\nTeacher train accuracy:", teacher_train_acc)
    print("Teacher validation accuracy:", teacher_val_acc)
    print("Teacher test accuracy:", teacher_test_acc)

    teacher.trainable = False

    # ============================================================
    # 2. Train student model using knowledge distillation
    # ============================================================
    student = build_student_model(input_shape, num_classes)

    print("\n===== Student Model Summary =====")
    student.summary()

    distiller = Distiller(student=student, teacher=teacher)

    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(),
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=ALPHA,
        temperature=TEMPERATURE,
    )

    print("\n===== Distillation Training =====")
    student_history = distiller.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    model = student

    # Compile the student model before using evaluate() or predict()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    plot_training_curves(student_history, "./images")

    # ============================================================
    # 3. Evaluate final student model
    # ============================================================
    train_loss, train_acc = model.evaluate(train_ds, verbose=0)
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    print("\nStudent train accuracy:", train_acc)
    print("Student validation accuracy:", val_acc)
    print("Student test accuracy:", test_acc)

    y_true, y_pred = get_predictions(model, test_ds)

    cm = tf.math.confusion_matrix(
        y_true,
        y_pred,
        num_classes=num_classes
    ).numpy()

    print("\nConfusion Matrix:")
    print(cm)

    plot_confusion_matrix(
        cm,
        LABELS,
        "./images/confusion_matrix.png"
    )

    save_confusion_matrix_csv(
        cm,
        LABELS,
        "./results/confusion_matrix.csv"
    )

    backward_id = LABELS.index("backward")
    custom_id = LABELS.index(CUSTOM_WORD)

    backward_frr = calculate_frr(y_true, y_pred, backward_id)
    custom_frr = calculate_frr(y_true, y_pred, custom_id)

    print("\nFalse Rejection Rate:")
    print("FRR for backward:", backward_frr)
    print(f"FRR for {CUSTOM_WORD}:", custom_frr)

    teacher_params = teacher.count_params()
    student_params = student.count_params()

    teacher_macs = estimate_macs(teacher)
    student_macs = estimate_macs(student)

    print("\nModel Summary Values:")
    print("Teacher parameters:", teacher_params)
    print("Student parameters:", student_params)
    print("Teacher estimated MACs:", teacher_macs)
    print("Student estimated MACs:", student_macs)
    print("Input tensor shape:", input_shape)

    # ============================================================
    # 4. Save models
    # ============================================================
    TEACHER_H5_MODEL = "../final_model/teacher_keyword_model.h5"
    STUDENT_H5_MODEL = "../final_model/student_keyword_model.h5"
    DEPLOY_H5_MODEL = "../final_model/keyword_model.h5"
    TFLITE_MODEL = "../final_model/keyword_model_int8.tflite"

    teacher.save(TEACHER_H5_MODEL)
    student.save(STUDENT_H5_MODEL)
    model.save(DEPLOY_H5_MODEL)

    print("\nSaved teacher model:", TEACHER_H5_MODEL)
    print("Saved student model:", STUDENT_H5_MODEL)
    print("Saved deployable model:", DEPLOY_H5_MODEL)

    # ============================================================
    # 5. Convert student model to full INT8 TFLite
    # ============================================================
    input_details, output_details = convert_to_full_int8_tflite(
        model,
        train_ds,
        TFLITE_MODEL
    )

    tflite_input_shape = input_details[0]["shape"]
    tflite_output_shape = output_details[0]["shape"]
    tflite_input_dtype = input_details[0]["dtype"]
    tflite_output_dtype = output_details[0]["dtype"]

    # ============================================================
    # 6. Save summary table
    # ============================================================
    with open("./results/summary_table.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["Metric", "Value"])

        writer.writerow(["Teacher Train Accuracy", teacher_train_acc])
        writer.writerow(["Teacher Validation Accuracy", teacher_val_acc])
        writer.writerow(["Teacher Test Accuracy", teacher_test_acc])

        writer.writerow(["Student Train Accuracy", train_acc])
        writer.writerow(["Student Validation Accuracy", val_acc])
        writer.writerow(["Student Test Accuracy", test_acc])

        writer.writerow(["FRR for Speech Commands word backward", backward_frr])
        writer.writerow([f"FRR for custom word {CUSTOM_WORD}", custom_frr])

        writer.writerow(["Teacher Number of Parameters", teacher_params])
        writer.writerow(["Student Number of Parameters", student_params])

        writer.writerow(["Teacher Estimated MACs", teacher_macs])
        writer.writerow(["Student Estimated MACs", student_macs])

        writer.writerow(["Keras Input Tensor Shape", str(tuple(input_shape))])
        writer.writerow(["TFLite Input Tensor Shape", str(tflite_input_shape)])
        writer.writerow(["TFLite Output Tensor Shape", str(tflite_output_shape)])

        writer.writerow(["TFLite Input Dtype", str(tflite_input_dtype)])
        writer.writerow(["TFLite Output Dtype", str(tflite_output_dtype)])

        writer.writerow(["Knowledge Distillation Temperature", TEMPERATURE])
        writer.writerow(["Knowledge Distillation Alpha", ALPHA])

        writer.writerow(["Sampling Rate", SAMPLE_RATE])
        writer.writerow(["Clip Duration", CLIP_DURATION])
        writer.writerow(["Samples Per Clip", SAMPLES_PER_CLIP])

        writer.writerow(["Frame Length for STFT", 256])
        writer.writerow(["Frame Step for STFT", 128])

        writer.writerow(["False Alarm Rate FA/hour", "Measured on board / streaming test"])

    print("\nSaved results:")
    print("./images/loss_curve.png")
    print("./images/accuracy_curve.png")
    print("./images/confusion_matrix.png")
    print("./results/confusion_matrix.csv")
    print("./results/summary_table.csv")
    print(TEACHER_H5_MODEL)
    print(STUDENT_H5_MODEL)
    print(DEPLOY_H5_MODEL)
    print(TFLITE_MODEL)

    print("\nNext step:")
    print("Run xxd -i ../final_model/keyword_model_int8.tflite > model_data.cc")
    print("Then copy model_data.cc and model_data.h into your Arduino sketch folder.")


if __name__ == "__main__":
    main()