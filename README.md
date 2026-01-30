# Sentiment Analysis with BOW, DAN, and BPE (SUBWORDDAN)

This repository contains code for sentiment analysis using Bag-of-Words (BOW), Deep Averaging Network (DAN), and Subword-based DAN (BPE) models. The code is implemented in Python with PyTorch.

## Requirements

- Python 3.8+
- PyTorch
- matplotlib

Install dependencies (if not already installed):
```
pip install torch matplotlib
```

## Data

Place your data files in the `data/` directory:
- `train.txt` — Training data (label \t sentence)
- `dev.txt` — Development/validation data (label \t sentence)
- `glove.6B.300d-relativized.txt` — GloVe embeddings (300d, relativized)

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
- All BPE segmentation is performed in memory (no intermediate files).
- For best results, use the provided GloVe file and ensure data is formatted as expected.

## Troubleshooting
- If you encounter slow BPE processing, try reducing `num_merges` for testing.
- Ensure all dependencies are installed and data files are present in the `data/` directory.

## License
This project is for educational purposes.
