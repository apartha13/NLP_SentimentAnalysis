# Sentiment Analysis with BOW, DAN, and BPE (SUBWORDDAN)

This repository contains code for sentiment analysis using Bag-of-Words (BOW), Deep Averaging Network (DAN), and Subword-based DAN (BPE) models. Code is implemented in Python with PyTorch.

## Requirements

- Python 3.8+
- PyTorch
- matplotlib

Install dependencies (if not already installed):
```
pip install torch matplotlib
```
Ideally, create a virtual environment to install these dependencies as seen in the PDF for the assignment.

## Data

Place your data files in the `data/` directory:
- `train.txt` — Training data (label \t sentence)
- `dev.txt` — Development/validation data (label \t sentence)
- `glove.6B.300d-relativized.txt` — GloVe embeddings (300d, relativized). 300d was more optimal than 50d as it allowed us to maximize our dev accuracy.

## Running the Code

All experiments are run via `main.py` using the `--model` argument:

### 1. Bag-of-Words (BOW)
```
python main.py --model BOW
```

### 2. Deep Averaging Network (DAN)
```
python main.py --model DAN
```

### 3. Subword DAN (BPE)
```
python main.py --model SUBWORDDAN
```

- Results and plots will be saved as PNG files in the working directory.
- You can adjust BPE vocabulary size by changing `num_merges` in `main.py` (SUBWORDDAN section).

## File Structure
- `main.py` — Main experiment script
- `BOWmodels.py` — BOW and neural BOW models
- `DANmodels.py` — DAN model and utilities
- `bpe_utils.py` — BPE implementation and dataset
- `sentiment_data.py`, `utils.py` — Additional utilities
- `data/` — Data and embeddings

## Notes
- If you want the best results, use the provided GloVe file and ensure data is formatted as expected.

## Troubleshooting
- If you encounter slow BPE processing, try reducing `num_merges` for testing but the one I used is 10,000 merges which is the optimal vocab_size for me.
- Ensure all dependencies are installed and data files are present in the `data/` directory.
