import os
import pathlib
import numpy as np
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

SAMPLE_RATE = 16000
CLIP_DURATION = 1.0
SAMPLES_PER_CLIP = int(SAMPLE_RATE * CLIP_DURATION)

DATA_DIR = pathlib.Path("data")
SILENCE_DIR = DATA_DIR / "silence"

NUM_SILENCE_SAMPLES = 1000


def save_wav(file_path, audio):
    audio = np.clip(audio, -1.0, 1.0)
    audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
    audio_tensor = tf.expand_dims(audio_tensor, axis=-1)

    wav_binary = tf.audio.encode_wav(audio_tensor, SAMPLE_RATE)
    tf.io.write_file(str(file_path), wav_binary)


def generate_gaussian_silence():
    # Very small Gaussian noise to simulate microphone/background noise
    noise_std = np.random.uniform(0.001, 0.01)
    audio = np.random.normal(
        loc=0.0,
        scale=noise_std,
        size=SAMPLES_PER_CLIP
    )

    return audio.astype(np.float32)


def add_low_frequency_hum(audio):
    # Optional low-frequency sine wave to simulate fan/AC/electrical hum
    t = np.arange(SAMPLES_PER_CLIP) / SAMPLE_RATE

    frequency = np.random.uniform(50, 200)
    amplitude = np.random.uniform(0.001, 0.006)

    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)

    return audio + sine_wave


def apply_random_gain(audio):
    gain = np.random.uniform(0.5, 1.5)
    return audio * gain


def generate_one_silence_clip():
    audio = generate_gaussian_silence()

    # Add low-frequency hum to some samples
    if np.random.rand() < 0.5:
        audio = add_low_frequency_hum(audio)

    audio = apply_random_gain(audio)

    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def main():
    os.makedirs(SILENCE_DIR, exist_ok=True)

    for i in range(NUM_SILENCE_SAMPLES):
        audio = generate_one_silence_clip()

        file_name = f"silence_{i:04d}.wav"
        file_path = SILENCE_DIR / file_name

        save_wav(file_path, audio)

        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{NUM_SILENCE_SAMPLES} silence samples")

    print("Done.")
    print("Silence files saved to:", SILENCE_DIR)


if __name__ == "__main__":
    main()