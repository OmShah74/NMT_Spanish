# Advanced Neural Machine Translation: Spanish to English with Transformers

This project implements a complete, state-of-the-art pipeline for training and evaluating a Neural Machine Translation (NMT) model to translate text from Spanish to English. The architecture is based on the powerful **Transformer model**, as introduced in the paper "Attention Is All You Need".

The system incorporates an advanced, linguistically-aware data preprocessing pipeline and supports sophisticated decoding strategies like Beam Search, making it a robust and high-performance solution. The model is trained on the Europarl v7 dataset.

## Core Features

-   **State-of-the-Art Transformer Architecture:** Utilizes the canonical Transformer model, relying entirely on multi-head self-attention to capture global dependencies in text.
-   **Advanced NLP Preprocessing:**
    -   **Linguistic Normalization:** Employs `spaCy` for lemmatization, reducing words to their root forms to consolidate the vocabulary and improve generalization.
    -   **Subword Tokenization:** Uses Google's `SentencePiece` to train a Byte Pair Encoding (BPE) model. This virtually eliminates the "unknown word" problem and allows the model to handle a large and open vocabulary efficiently.
-   **Sophisticated Decoding:** Implements both a fast **Greedy Search** for quick inference and a more powerful **Beam Search** algorithm to generate higher-quality translations for final evaluation.
-   **GPU & Mixed Precision Support:** The training script automatically detects and uses a CUDA-enabled GPU. It leverages Automatic Mixed Precision (AMP) to accelerate training and reduce memory usage.
-   **Resumable Training:** The training script is designed to be resumable. If interrupted, it can load the last best-saved model and continue training, saving significant time.
-   **Comprehensive Evaluation:** Measures performance using standard metrics, including Cross-Entropy Loss, Perplexity, and the industry-standard **BLEU score**.
-   **Modular and Clean Codebase:** The project is structured with clear, separate Python modules for configuration, data processing, model architecture, training, testing, and inference.

## Architecture and Pipeline Overview

The project follows a modern, end-to-end NMT pipeline:

1.  **Data Preprocessing:** Raw Spanish and English text is first cleaned and normalized using `spaCy` to lemmatize words. Then, a `SentencePiece` tokenizer is trained on this cleaned text to create a subword vocabulary. This two-step process produces a robust representation that handles morphological variations and rare words.

2.  **Model - The Transformer:** The architecture consists of an Encoder and a Decoder, but without any recurrent layers.
    *   **Encoder:** A stack of Transformer layers processes the entire source sentence in parallel. Each layer uses multi-head self-attention to build a context-rich representation for every subword token.
    *   **Decoder:** Another stack of Transformer layers generates the translation token by token. At each step, it attends to the previously generated tokens and the final output of the encoder to predict the next best subword.

3.  **Inference (Decoding):**
    *   **Greedy Search:** At each step, the model selects the single most probable next token. It's fast but can be suboptimal.
    *   **Beam Search:** At each step, the model keeps track of the `k` most probable partial translations (the "beam") and expands them, leading to a more globally optimal and fluent final translation.

## Setup and Installation

Follow these steps to set up the project environment on your local machine or in a cloud environment like Kaggle/Colab.

### Prerequisites

-   Python 3.9+
-   Git
-   (Recommended for Training) An NVIDIA GPU with CUDA support.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd NMT
    ```

2.  **Create and activate a virtual environment (highly recommended):**
    ```bash
    python -m venv .venv
    
    # On Windows (PowerShell as Administrator)
    Set-ExecutionPolicy RemoteSigned -Scope Process
    .\.venv\Scripts\activate
    
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLP models for spaCy:**
    ```bash
    python -m spacy download en_core_web_sm
    python -m spacy download es_core_news_sm
    ```

## Usage: The Full Pipeline

Execute the following commands from the root `NMT/` directory. **Always use the `-m` flag** to run the scripts as modules.

### Step 1: Preprocess the Data

This script will load the raw Europarl dataset, perform lemmatization, train the SentencePiece tokenizers, and save all processed files into the `data/processed/` directory.

```bash
python -m src.data_preprocessing
```

This step can be time-consuming due to the spaCy processing.

### Step 2: Train the Transformer Model

This command starts the training process. The script will automatically detect a GPU if available. The best-performing model (based on validation loss) will be saved to saved_models/best-transformer-model.pt. If the script is stopped and restarted, it will automatically resume from this saved model.

```bash
python -m src.train
```

### Step 3: Evaluate the Model on the Test Set

After training is complete, run this script to evaluate the best model on the unseen test set. It will report the final BLEU score using Beam Search decoding.

```bash
python -m src.test
```
The script will also print a few example translations for a qualitative check.

### Step 4: Translate a New Sentence

Use the translate.py script to perform inference on any new Spanish sentence. By default, it uses the efficient Beam Search algorithm.

```bash
python -m src.translate "la cooperación es la clave del éxito"
```

Example Output:
```code
--- Translating with Beam Search ---
Original (es): la cooperación es la clave del éxito
Translated (en): cooperation is the key to success
```