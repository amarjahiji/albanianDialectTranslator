# Albanian Dialect Translator

This repository contains a university project developed for the **Intro to Artificial Intelligence** course. The project implements an end-to-end translation system that converts Gheg dialect Albanian into Standard Albanian using a context-aware Transformer model built with PyTorch.


## Overview

The translation system is built around a custom Transformer architecture that:
- **Learns** a vocabulary directly from the dataset.
- **Translates** sentences using beam search for improved accuracy.
- **Evaluates** translations using industry-standard metrics such as BLEU Score, Character Error Rate (CER), and Word Error Rate (WER).

Additionally, a Streamlit-based web interface is provided for interactive translation and evaluation.

## Project Structure

```plaintext
.
├── dataset.py          # Contains the training sentence pairs
├── model.py            # Defines the Transformer model, training loop, and translation function
├── test.py             # Streamlit app for interactive translation and evaluation
├── best_model.pt       # Saved model weights (generated after training)
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation (this file)

## Installation

1. **Clone the Repository:**
   ```bash
   git clone <your-repo-url>
   cd <repository-folder>

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt


## Training the Model

To train the model, run the model.py script. This script will:
    - Initialize the Transformer model.
    - Train the model for up to 100 epochs with early stopping based on average loss.
    - Save the best performing model as best_model.pt.
Run the training process with:
    ```bash
    python model.py


## Running the Translation and Evaluation Interface

A Streamlit web application is provided for interactive translation and evaluation. To launch the app:
    ```bash
    streamlit run test.py
After running the command, open the provided local URL in your browser. The app allows you to:
    - Translate input sentences from the Gheg dialect to Standard Albanian.
    - Evaluate the translations with BLEU, CER, and WER metrics.
    - View sample translations and aggregate evaluation results.


## Evaluation Metrics

The following metrics are used to evaluate translation quality:
**BLEU Score:**
    Measures n-gram overlap between the predicted translation and the reference.
    Higher is better (range: 0 to 1).
**Character Error Rate (CER):**
    Computes the normalized edit distance at the character level between the predicted translation and the reference.
    Lower is better (range: 0 to 1).
**Word Error Rate (WER):**
    Calculates the normalized edit distance at the word level.
    Lower is better (range: 0 to 1).


## Acknowledgments

This project was developed as part of the Introduction to Artificial Intelligence course and leverages modern techniques in neural machine translation using PyTorch and Streamlit.
