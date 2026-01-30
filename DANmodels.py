import torch
from torch import nn
import torch.nn.functional as F
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset
import numpy as np


class SentimentDatasetDAN(Dataset):
    """
    This is a PyTorch Dataset for word-level DAN models. This allows me to convert sentences
    into sequences that are padded to a fixed length, and convert words to indices.
    """
    def __init__(self, infile, word_to_idx=None, train=True, max_seq_length=100):
        # Reads sentiment examples from the input files given to us in the data folder. 
        self.examples = read_sentiment_examples(infile)
        self.max_seq_length = max_seq_length
        self.word_to_idx = word_to_idx
        
        # If there is no vocab provided, we build it from the training data
        if train and word_to_idx is None:
            self.word_to_idx = self._build_vocab()
        
    def _build_vocab(self):
        # This function allowed me to construct a mapping from words to indices
        word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        idx = 2

        for example in self.examples:
            for word in example.words:
                if word not in word_to_idx:
                    word_to_idx[word] = idx
                    idx += 1
        
        return word_to_idx
    
    def _text_to_indices(self, words):
        # Converts list of words to corresponding indices
        indices = []
        for word in words:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx["<UNK>"])
        
        # Pad/truncate
        if len(indices) < self.max_seq_length:
            indices += [self.word_to_idx["<PAD>"]] * (self.max_seq_length - len(indices))
        else:
            indices = indices[:self.max_seq_length]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # Returns the indices tensor and label for index
        example = self.examples[idx]
        indices = self._text_to_indices(example.words)
        label = torch.tensor(example.label, dtype=torch.long)
        return indices, label

class DAN_2Layer(nn.Module):
    """
    2-layer DAN supporting GloVe, random, and BPE embeddings. Used for sentiment classification.
    """
    def __init__(self, pretrained_embeddings=None, vocab_size=None, embedding_dim=100, hidden_size=100, dropout=0.3, emb_dropout=0.0, freeze_embeddings=False):
        super().__init__()
        # Initialize embeddings based on whether pretrained embeddings are provided
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze_embeddings, padding_idx=0)
            embedding_dim = pretrained_embeddings.shape[1]
        else:
            assert vocab_size is not None, "vocab_size must be provided if not using pretrained embeddings"
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Dropout for embeddings and hidden layer
        # Also, having a linear layer from embedding_dim to hidden_size and then an output layer
        self.emb_dropout = nn.Dropout(emb_dropout) if emb_dropout > 0 else nn.Identity()
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # Embed input indices and apply the dropout for embedds
        embedded = self.embedding(x)
        embedded = self.emb_dropout(embedded)

        # masking out the pad tokens so we can average. Then averaging
        mask = (x != 0).float()
        mask = mask.unsqueeze(-1)
        summed = (embedded * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        averaged = summed / lengths

        # Passing through hidden layers and output log probs for the classes
        hidden = F.relu(self.fc1(averaged))
        hidden = self.dropout1(hidden)
        output = F.log_softmax(self.fc2(hidden), dim=1)
        return output


def load_glove_embeddings(glove_file, word_to_idx):
    """
    Loads GloVe embeddings for the vocabulary.
    Then returns the embedding matrix and embedding dimension.
    """
    # First pass: determne embedding dimension
    embedding_dim = None
    with open(glove_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().split()
        embedding_dim = len(first_line) - 1  # -1 for the word itself
    
    print(f"Detected embedding dimension: {embedding_dim}D")
    
    # Initialize embeddings matrix with rand vals
    embeddings = np.random.randn(len(word_to_idx), embedding_dim) * 0.1
    embeddings[0] = 0  # PAD token gets zero vector
    
    # Second pass: fill embedding matrix with GloVe vectors
    loaded_count = 0
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word_to_idx:
                vector = np.array([float(x) for x in values[1:]], dtype=np.float32)
                embeddings[word_to_idx[word]] = vector
                loaded_count += 1
    
    print(f"Loaded {loaded_count} embeddings from {glove_file}")
    return torch.from_numpy(embeddings).float(), embedding_dim