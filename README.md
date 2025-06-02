# Introduction

This is the README for running the Deep Learning–Based Methods developed in the thesis project **“Real-Time Unsupervised Anomaly Detection for Manufacturing Quality Inspection.”** In this work, I experiment with multiple unsupervised learning methods to detect defects in manufacturing (using a concrete-crack dataset as a proxy) and compare them against a supervised benchmark. The goal is to find an unsupervised approach that requires only normal (defect-free) data yet still identifies anomalies in real time.

The repository includes training code for the following models:
- **Convolutional Autoencoder (CAE)**
- **EfficientNet-Based Autoencoder (EfficientAE)**
- **Patch Description Network (PDN)** (Knowledge Distillation from a WideResNet-101 teacher to a lightweight 4-layer student)
- **Custom 5-Layer CNN Baseline** (supervised)
- **ResNet-18 (Transfer Learning) Baseline** (supervised)

# Getting Started

## Environment

- **Hardware**: GPU recommended (NVIDIA T4 or equivalent). Training and inference will run on CPU, but GPU accelerates both model training and evaluation.
- **Python version**: 3.9 (tested) or later

## Dependencies

Install the required packages (tested on Python 3.9):

```bash
pip install --upgrade pip
pip install torch torchvision numpy pandas pillow scikit-learn matplotlib kornia tqdm tensorboard
