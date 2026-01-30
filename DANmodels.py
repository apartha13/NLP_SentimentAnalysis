import torch
from torch import nn
import torch.nn.functional as F
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset
import numpy as np


class SentimentDatasetDAN(Dataset):
    """
    Dataset class for DAN models that converts sentences to word indices
    """
    def __init__(self, infile, word_to_idx=None, train=True, max_seq_length=100):
        self.examples = read_sentiment_examples(infile)
        self.max_seq_length = max_seq_length
        self.word_to_idx = word_to_idx
        
        # Build vocabulary from training data
        if train and word_to_idx is None:
            self.word_to_idx = self._build_vocab()
        
    def _build_vocab(self):
        """Build vocabulary from examples"""
        word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        idx = 2
        for example in self.examples:
            for word in example.words:
                if word not in word_to_idx:
                    word_to_idx[word] = idx
                    idx += 1
        return word_to_idx
    
    def _text_to_indices(self, words):
        """Convert words to indices"""
        indices = []
        for word in words:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx["<UNK>"])
        
        # Pad or truncate
        if len(indices) < self.max_seq_length:
            indices += [self.word_to_idx["<PAD>"]] * (self.max_seq_length - len(indices))
        else:
            indices = indices[:self.max_seq_length]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        indices = self._text_to_indices(example.words)
        label = torch.tensor(example.label, dtype=torch.long)
        return indices, label



# ==================== Unified DAN 2-Layer Model ====================
class DAN_2Layer(nn.Module):
    """
    2-layer DAN supporting GloVe, random, or BPE embeddings.
    Use pretrained_embeddings for GloVe (set freeze_embeddings=True for frozen),
    or vocab_size/embedding_dim for random/BPE (set freeze_embeddings=False).
    """
    def __init__(self, pretrained_embeddings=None, vocab_size=None, embedding_dim=100, hidden_size=100, dropout=0.3, emb_dropout=0.0, freeze_embeddings=False):
        super().__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze_embeddings, padding_idx=0)
            embedding_dim = pretrained_embeddings.shape[1]
        else:
            assert vocab_size is not None, "vocab_size must be provided if not using pretrained embeddings"
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(emb_dropout) if emb_dropout > 0 else nn.Identity()
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.emb_dropout(embedded)

        mask = (x != 0).float()
        mask = mask.unsqueeze(-1)

        summed = (embedded * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        averaged = summed / lengths

        hidden = F.relu(self.fc1(averaged))
        hidden = self.dropout1(hidden)
        output = F.log_softmax(self.fc2(hidden), dim=1)
        return output


# ==================== Utility Functions ====================

def load_glove_embeddings(glove_file, word_to_idx):
    """
    Load GloVe embeddings and create embedding matrix for vocabulary.
    Automatically detects embedding dimension from the file.
    """
    # First pass: detect embedding dimension
    embedding_dim = None
    with open(glove_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().split()
        embedding_dim = len(first_line) - 1  # -1 for the word itself
    
    print(f"Detected embedding dimension: {embedding_dim}D")
    
    # Initialize embeddings with small random values
    embeddings = np.random.randn(len(word_to_idx), embedding_dim) * 0.1
    embeddings[0] = 0  # PAD token gets zero vector
    
    # Second pass: load embeddings
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