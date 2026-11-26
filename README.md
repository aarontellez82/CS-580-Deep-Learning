# CS-580-Deep-Learning
Deep learning implementations in Python with TensorFlow, Keras, PyTorch, and Hugging Face, covering CNN quantization, LSTM time series and text models, transformer fine-tuning, and multimodal classification.

Implementations, experiments, and empirical evaluations for **CS 580 (Deep Learning)**, built in Python using TensorFlow/Keras, PyTorch, and Hugging Face `transformers`.

## Overview

This repository documents neural network architectures, training workflows, and model evaluation across vision, sequential, natural language, and multimodal tasks—emphasizing shape verification, regularization, validation curve monitoring, and deployment trade-offs.

## Key Projects & Experiments

* **Post-Training Model Quantization (MNIST CNN):** Converted a Keras convolutional neural network (`2.61 MB`, `99.00%` accuracy) into a TensorFlow Lite model (`0.22 MB`, `98.38%` accuracy) to evaluate size and latency trade-offs.
* **Time-Series Forecasting (Jena Climate LSTM):** Built an LSTM sequence forecasting pipeline with temporal windowing and feature scaling, achieving a test MSE of `0.0066` and test MAE of `0.0573` over 15 epochs.
* **Text Classification (Bidirectional LSTM):** Developed a Bidirectional LSTM classifier for spam/ham detection, reaching `98.21%` test accuracy and a `0.9528` validation F1 score.
* **Sentiment Analysis (Fine-Tuned BERT):** Fine-tuned a pretrained BERT model on IMDb movie reviews using Hugging Face `Trainer`, comparing baseline and tuned hyperparameter configurations.
* **Bimodal Traffic Classification:** Evaluated deep learning models across *Normal*, *Accident*, *Road Hazard*, and *Severe Congestion* classes, tuning training duration to 3 epochs to prevent overfitting and achieve `91%` test accuracy.

## Tech Stack

* **Languages & Environments:** Python, Jupyter Notebooks, Linux / Windows (`uv`)
* **Deep Learning & NLP:** TensorFlow, Keras, PyTorch, Hugging Face `transformers`
* **Data & Evaluation:** NumPy, `pandas`, `scikit-learn`, `matplotlib`

## Repository Structure

```text
.
├── notebooks/          # Experimentation and training notebooks
├── src/                # Modular data prep, model, and evaluation scripts
├── artifacts/          # Saved models, quantized TFLite models, and plots
└── README.md
