# Neural Machine Translation: Spanish to English

This project implements a complete pipeline for training and evaluating a Neural Machine Translation (NMT) model to translate text from Spanish to English. The architecture is based on a Sequence-to-Sequence (Seq2Seq) model with a Bidirectional GRU Encoder and an Attention-based GRU Decoder. The model is trained on the Europarl v7 dataset.

## Features

-   **Modern NMT Architecture:** Utilizes an attention mechanism, allowing the model to focus on the most relevant parts of the source sentence while translating.
-   **Bidirectional Encoder:** The encoder processes the source text in both forward and backward directions to capture richer context.
-   **GPU & Mixed Precision Support:** The training script automatically detects and uses a CUDA-enabled GPU for accelerated training. It also supports Automatic Mixed Precision (AMP) to speed up training and reduce memory usage on compatible GPUs.
-   **Comprehensive Evaluation:** The model's performance is measured using standard metrics, including Cross-Entropy Loss, Perplexity, and the industry-standard **BLEU score**.
-   **Progress Logging:** Training and evaluation loops feature a `tqdm` progress bar, providing real-time feedback and ETA.
-   **Modular Codebase:** The project is structured with clear, separate modules for configuration, data processing, model architecture, training, testing, and inference.

## Architecture Overview

The model follows a classic Encoder-Decoder framework enhanced with an attention mechanism:

1.  **Encoder:** A multi-layer bidirectional Gated Recurrent Unit (GRU) reads the input Spanish sentence. Its goal is to produce two outputs:
    *   A set of **encoder outputs** for every word, capturing the contextual meaning of that word.
    *   A final **hidden state** (or "thought vector") that summarizes the entire sentence.

2.  **Attention Mechanism:** This component acts as a bridge between the encoder and decoder. At each step of the translation, it calculates "attention weights" that determine which of the encoder outputs are most relevant. This allows the decoder to focus its "attention" on specific Spanish words when generating the corresponding English word.

3.  **Decoder:** A multi-layer GRU that generates the English translation one word at a time. At each step, it takes the previously generated word and a context vector (a weighted sum of encoder outputs, provided by the attention mechanism) to predict the next word in the sequence.

## Setup and Installation

Follow these steps to set up the project environment.

### Prerequisites

-   Python 3.8+
-   Git
-   (Optional but Recommended) An NVIDIA GPU with CUDA support for training.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd NMT
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage: The Full Pipeline

Execute the following commands from the root `NMT/` directory.

### Step 1: Preprocess the Data

This script will load the raw Europarl dataset, normalize the text, build vocabularies, and save the processed data splits into the `data/processed/` directory.

```bash
python -m src.data_preprocessing
Step 2: Train the Model
This command starts the training process. The script will automatically detect if a GPU is available. The best performing model (based on validation loss) will be saved to saved_models/best-nmt-model.pt.
code
Bash
python -m src.train
You will see a progress bar for each epoch, showing the training loss, speed, and ETA.
Step 3: Evaluate the Model on the Test Set
After training is complete, run this script to evaluate the best model on the unseen test set. It will report the final Loss, Perplexity, and BLEU score.
code
Bash
python -m src.test
The script will also print a few example translations for a qualitative check.
Step 4: Translate a New Sentence
Use the translate.py script to perform inference on a single Spanish sentence.
code
Bash
python -m src.translate "este es un ejemplo de una frase en español"
Example Output:
code
Original (es): este es un ejemplo de una frase en español
Translated (en): this is an example of a sentence in spanish
Running on Google Colab
This project is well-suited for Google Colab's free T4 GPU.
Prepare: Compress the entire NMT project folder into a NMT.zip file.
Setup Notebook: In a new Colab notebook, set the runtime to T4 GPU (Runtime -> Change runtime type).
Upload and Unzip: Use the following code in a cell to upload and extract your project.
code
Python
from google.colab import files
uploaded = files.upload() # Upload NMT.zip
!unzip NMT.zip
Install and Run: Navigate into the project directory and run the commands as described in the "Usage" section.
code
Python
import os
os.chdir('NMT')

# Install dependencies
!pip install -r requirements.txt

# Run the pipeline
!python src/data_preprocessing.py
!python src/train.py
!python src/test.py
!python src/translate.py "hola mundo"