# Instrument-Classification

# Musical Instrument Classification using CNN

A deep learning system that classifies musical instruments from audio clips using Convolutional Neural Networks (CNNs) trained on mel-spectrogram representations. The model achieves **97.23% validation accuracy** across 28 instrument classes.

---

## Overview

This project converts raw audio files into mel-spectrograms and trains a custom CNN to recognize which musical instrument is being played. The pipeline handles everything from data preprocessing to real-time inference on new audio files.

**Key highlights:**
- 28 instrument classes, 42,311 total audio samples
- Custom 4-block CNN with batch normalization and dropout
- Mel-spectrogram preprocessing with SpecAugment-style data augmentation
- Full inference pipeline supporting `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`

---

## Dataset

| Split | Samples | Percentage |
|-------|---------|------------|
| Train | 29,617 | 70% |
| Validation | 4,231 | 10% |
| Test | 8,463 | 20% |

Splits are **stratified** to preserve class distribution. The dataset contains significant class imbalance (Harmonica: 131 samples vs. Flute: 3,719 samples), which is handled through stratified splitting and spectrogram augmentation during training.

### Instrument Classes (28)

| # | Instrument | Samples | # | Instrument | Samples |
|---|------------|---------|---|------------|---------|
| 1 | Accordion | 3,581 | 15 | Keyboard | 2,041 |
| 2 | Acoustic Guitar | 3,654 | 16 | Mandolin | 2,458 |
| 3 | Banjo | 2,998 | 17 | Organ | 1,442 |
| 4 | Bass Guitar | 3,613 | 18 | Piano | 575 |
| 5 | Clarinet | 634 | 19 | Saxophone | 454 |
| 6 | Cymbals | 208 | 20 | Shakers | 1,357 |
| 7 | Dobro | 487 | 21 | Tambourine | 558 |
| 8 | Drum Set | 3,648 | 22 | Trombone | 2,965 |
| 9 | Electric Guitar | 1,316 | 23 | Trumpet | 503 |
| 10 | Floor Tom | 406 | 24 | Ukulele | 790 |
| 11 | Harmonica | 131 | 25 | Violin | 630 |
| 12 | Harmonium | 1,314 | 26 | Cowbell | 621 |
| 13 | Hi-Hats | 444 | 27 | Flute | 3,719 |
| 14 | Horn | 1,258 | 28 | Vibraphone | 506 |

---

## Project Structure

instruments/
├── instrument_classification.ipynb   # Full implementation notebook
├── best_instrument_model.pth         # Trained model checkpoint (~5.1 MB)
├── test_visualization.png            # Mel-spectrogram sample visualization
├── mel_spectrograms/                 # Pre-computed mel-spectrograms
│   ├── Accordion/
│   │   ├── 3109.npy
│   │   └── ...
│   ├── Acoustic_Guitar/
│   └── ... (28 instrument directories)
├── IMG_5562.wav                      # Sample audio for testing
├── piano-classical-music-347514.mp3  # Piano sample
├── simple-guitar-melody-102329.mp3   # Guitar sample
└── piano-chords-239967.mp3           # Piano chords sample



---

## Audio Preprocessing

Raw audio files are converted to mel-spectrograms and saved as `.npy` arrays before training. This avoids recomputing spectrograms on every epoch.

**Audio parameters:**

| Parameter | Value |
|-----------|-------|
| Sample rate | 22,050 Hz |
| Clip duration | 3 seconds |
| Target samples | 66,150 (22050 × 3) |
| Mel bands (n_mels) | 128 |
| FFT window (n_fft) | 2,048 |
| Hop length | 512 |

**Preprocessing steps per audio file:**
1. Load audio with librosa
2. Resample to 22,050 Hz if needed
3. Convert stereo to mono
4. Pad (with zeros) or trim to exactly 3 seconds
5. Compute mel-spectrogram
6. Convert amplitude to dB scale (`power_to_db`)
7. Z-score normalize: `(mel_spec - mean) / (std + 1e-9)`
8. Save as `.npy` to `mel_spectrograms/{class}/{id}.npy`

---

## Model Architecture

**InstrumentCNN** — a custom 4-block convolutional network with global average pooling.

Input: (batch, 1, 128, time_bins)

Conv Block 1:  Conv2d(1→32)   → BatchNorm → ReLU → MaxPool2d(2×2)
Conv Block 2:  Conv2d(32→64)  → BatchNorm → ReLU → MaxPool2d(2×2)
Conv Block 3:  Conv2d(64→128) → BatchNorm → ReLU → MaxPool2d(2×2)
Conv Block 4:  Conv2d(128→256)→ BatchNorm → ReLU → MaxPool2d(2×2)

Global Average Pooling: AdaptiveAvgPool2d(1, 1)  →  (batch, 256)

Classifier:
Dropout(0.5) → Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→28)

Output: (batch, 28)  — raw logits for CrossEntropyLoss



All convolutions use kernel size 3×3 with padding=1 to preserve spatial dimensions before pooling.

---

## Training

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |
| LR scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |

### Data Augmentation (training set only)

**SpecAugment-style masking** applied with 50% probability each:
- **Time masking:** Randomly zero out 0–20 consecutive time frames (replaced with spectrogram mean)
- **Frequency masking:** Randomly zero out 0–20 consecutive mel frequency bins (replaced with spectrogram mean)

### Training Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.9543 | 72.15% | 0.4764 | 85.37% |
| 2 | 0.5132 | 84.28% | 0.4274 | 86.81% |
| 3 | 0.3959 | 88.07% | 0.2616 | 91.42% |
| 4 | 0.3227 | 90.10% | 0.1904 | 94.26% |
| 5 | 0.2692 | 91.77% | 0.1401 | 95.93% |
| 8 | 0.1867 | 94.15% | 0.1063 | 96.74% |
| 10 | 0.1539 | 95.11% | 0.0971 | **97.23%** |

Best model (epoch 10) is automatically saved to `best_instrument_model.pth`.

The model checkpoint contains:
- Model state dictionary
- Optimizer state dictionary
- Class names and index mapping
- Best validation accuracy
- Epoch number

---

## Installation

```bash
pip install torch torchaudio librosa matplotlib numpy scikit-learn tqdm
Python 3.8+ is recommended. GPU acceleration is supported automatically if CUDA is available.

Usage
Step 1 — Preprocess audio to mel-spectrograms

converter = AudioConverter(
    sample_rate=22050,
    duration=3,
    n_mels=128,
    n_fft=2048,
    hop_length=512
)

converter.convert_dataset(
    input_dir='/path/to/raw_audio_dataset',
    output_dir='mel_spectrograms',
    save_format='npy'
)
The input directory should be organized as:


raw_audio_dataset/
├── Piano/
│   ├── sample1.wav
│   └── ...
├── Violin/
└── ...
Step 2 — Train the model

model, history, classes, class_to_idx = train_instrument_classifier(
    mel_spec_dir='mel_spectrograms',
    num_epochs=50,
    batch_size=32,
    lr=0.001
)
Step 3 — Run inference on a new audio file

predictor = InstrumentPredictor(
    model_path='best_instrument_model.pth',
    device='cpu'   # or 'cuda'
)

results = predictor.predict_and_print('path/to/audio.wav', top_k=3)
Example output:


============================================================
PREDICTION RESULTS
============================================================
Audio: piano-classical-music-347514.mp3

1. Piano               92.15% ████████████████████████████
2. Keyboard             5.23% █
3. Organ                2.62%

============================================================

Detected Instrument: PIANO
Confidence: 92.15%
Inference Pipeline
When predict() is called on a raw audio file, the following happens internally:

Load audio file (supports .wav, .mp3, .flac, .ogg, .m4a)
Resample to 22,050 Hz
Convert to mono
Pad or trim to 3 seconds
Compute mel-spectrogram (same parameters as training)
Convert to dB scale and z-score normalize
Run through CNN
Apply softmax to get per-class probabilities
Return top-k predictions
Requirements
Library	Purpose
torch	CNN model, training loop
torchaudio	Audio I/O
librosa	Mel-spectrogram computation
numpy	Array operations, .npy storage
matplotlib	Spectrogram visualization
scikit-learn	Stratified train/val/test split
tqdm	Training progress bars

