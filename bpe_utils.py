import re
from collections import Counter
import torch
from torch.utils.data import Dataset

class BPE:
    def __init__(self, num_merges=1000):
        self.num_merges = num_merges
        self.bpe_codes = []
        self.vocab = None
        self.bpe_ranks = {}

    def get_vocab(self, corpus):
        vocab = Counter()
        for line in corpus:
            for word in line.strip().split():
                chars = ' '.join(list(word)) + ' </w>'
                vocab[chars] += 1
        return vocab

    def get_stats(self, vocab):
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        pattern = re.escape(' '.join(pair))
        re_pattern = re.compile(r'(?<!\S)' + pattern + r'(?!\S)')
        new_vocab = {}
        for word in vocab:
            new_word = re_pattern.sub(''.join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def fit(self, corpus):
        self.vocab = self.get_vocab(corpus)
        for i in range(self.num_merges):
            pairs = self.get_stats(self.vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.vocab = self.merge_vocab(best, self.vocab)
            self.bpe_codes.append(best)
            if i % 100 == 0:
                print(f"BPE merge {i}: {best}")

        self.bpe_ranks = {pair: idx for idx, pair in enumerate(self.bpe_codes)}

    def encode_word(self, word):
        symbols = list(word) + ['</w>']
        while True:
            pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
            candidates = [(pair, self.bpe_ranks[pair]) for pair in pairs if pair in self.bpe_ranks]
            if not candidates:
                break
            best_pair = min(candidates, key=lambda x: x[1])[0]
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols)-1 and (symbols[i], symbols[i+1]) == best_pair:
                    new_symbols.append(symbols[i]+symbols[i+1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        if symbols[-1] == '</w>':
            symbols = symbols[:-1]
        return symbols

    def encode_sentence(self, sentence):
        return [self.encode_word(word) for word in sentence.strip().split()]

class BPESentimentDataset(Dataset):
    """
    PyTorch Dataset for BPE-segmented sentiment data (on-the-fly, no .bpe.txt needed).
    Each line in the input file should be: label\t<sentence>
    """
    def __init__(self, infile, bpe: BPE, word_to_idx=None, train=True, max_seq_length=100):
        self.examples = []
        self.max_seq_length = max_seq_length
        self.word_to_idx = word_to_idx
        self.bpe = bpe
        with open(infile, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line.strip()) == 0:
                    continue
                if '\t' in line:
                    label, sent = line.strip().split('\t', 1)
                    bpe_tokens = sum(self.bpe.encode_sentence(sent), [])
                    self.examples.append((bpe_tokens, int(label)))
                else:
                    bpe_tokens = sum(self.bpe.encode_sentence(line.strip()), [])
                    self.examples.append((bpe_tokens, None))
        # Build subword vocab from BPE tokens only
        if train and word_to_idx is None:
            self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
            idx = 2
            for tokens, _ in self.examples:
                for token in tokens:
                    if token not in self.word_to_idx:
                        self.word_to_idx[token] = idx
                        idx += 1
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        tokens, label = self.examples[idx]
        indices = [self.word_to_idx.get(tok, self.word_to_idx["<UNK>"]) for tok in tokens]
        if len(indices) < self.max_seq_length:
            indices += [self.word_to_idx["<PAD>"]] * (self.max_seq_length - len(indices))
        else:
            indices = indices[:self.max_seq_length]
        indices = torch.tensor(indices, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long) if label is not None else torch.tensor(-1, dtype=torch.long)
        return indices, label
