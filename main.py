from prometheus_client import Counter
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
import time
from collections import Counter
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import (
    SentimentDatasetDAN,
    DAN_2Layer,
    load_glove_embeddings
)

def train_epoch(data_loader, model, loss_fn, optimizer):
    """Train for one epoch (original signature/behavior)"""
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        # Decide dtype based on model/input: embedding models expect Long indices, BOW expects floats
        if hasattr(model, 'embedding') or X.dtype in (torch.long, torch.int):
            if X.dtype != torch.long:
                X = X.long()
        else:
            X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


def eval_epoch(data_loader, model, loss_fn, optimizer=None):
    """Evaluate on dev set (original signature/behavior)"""
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        # Match dtype logic from train_epoch
        if hasattr(model, 'embedding') or X.dtype in (torch.long, torch.int):
            if X.dtype != torch.long:
                X = X.long()
        else:
            X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


def experiment(model, train_loader, test_loader, epochs=100, lr=1e-4):
    """Compatibility wrapper — original experiment() behavior"""
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(epochs):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy


def main():
    parser = argparse.ArgumentParser(description='Train sentiment analysis models')
    parser.add_argument('--model', type=str, required=True, choices=['BOW', 'DAN', 'SUBWORDDAN'], 
                       help='Model type to train')
    args = parser.parse_args()

    # Use the 300d GloVe file always (hardcoded)
    glove_file = 'data/glove.6B.300d-relativized.txt'
    print(f"Using GloVe file: {glove_file}\n")

    # Load dataset (BOW defaults for convenience)
    start_time = time.time()
    train_data = SentimentDatasetBOW("data/train.txt")
    dev_data = SentimentDatasetBOW("data/dev.txt", vectorizer=train_data.vectorizer, train=False)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=16, shuffle=False)
    test_loader = dev_loader
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loaded in : {elapsed_time} seconds\n")

    # ==================== BOW Model ====================
    if args.model == "BOW":
        print("=" * 60)
        print("TRAINING BAG-OF-WORDS (BOW) MODELS")
        print("=" * 60)
        
        print("\nUsing preloaded BOW data...")
        
        # Train 2-layer BOW
        print("\n2-Layer BOW Network:")
        nn2_train_acc, nn2_dev_acc = experiment(
            NN2BOW(input_size=512, hidden_size=100),
            train_loader, test_loader
        )
        
        # Train 3-layer BOW
        print("\n3-Layer BOW Network:")
        nn3_train_acc, nn3_dev_acc = experiment(
            NN3BOW(input_size=512, hidden_size=100),
            train_loader, test_loader
        )
        
        # Plot results
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(nn2_train_acc, label='2-Layer Train')
        plt.plot(nn3_train_acc, label='3-Layer Train')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('BOW - Training Accuracy')
        plt.legend()
        plt.grid()
        
        plt.subplot(1, 2, 2)
        plt.plot(nn2_dev_acc, label='2-Layer Dev')
        plt.plot(nn3_dev_acc, label='3-Layer Dev')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('BOW - Dev Accuracy')
        plt.legend()
        plt.grid()
        
        plt.tight_layout()
        plt.savefig('bow_results.png', dpi=100)
        print("\n✓ Saved: bow_results.png")
    
    # ==================== DAN Model ====================
    elif args.model == "DAN":
        print("=" * 60)
        print("TRAINING DEEP AVERAGING NETWORK (DAN) MODELS")
        print("=" * 60)
        
        # Load data
        print("\nLoading data...")
        train_data = SentimentDatasetDAN("data/train.txt")
        dev_data = SentimentDatasetDAN("data/dev.txt", word_to_idx=train_data.word_to_idx, train=False)

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        dev_loader = DataLoader(dev_data, batch_size=16, shuffle=False)
                
        vocab_size = len(train_data.word_to_idx)
        print(f"Vocabulary size: {vocab_size}")
         
        # ========== PART 1A: GloVe Embeddings ==========
        print("\n" + "=" * 60)
        print("PART 1A: DAN with GloVe Embeddings (Frozen)")
        print("=" * 60)
        
        print(f"\nLoading GloVe embeddings from {glove_file}...")
        pretrained_embeddings, emb_dim = load_glove_embeddings(glove_file, train_data.word_to_idx)

        # Train final 2-layer model with best config (using experiment wrapper)
        print("\nTraining final 2-Layer GloVe model with best hyperparameters...")
        glove_2_train, glove_2_dev = experiment(
            DAN_2Layer(
                pretrained_embeddings=pretrained_embeddings,
                hidden_size=150,
                dropout=0.05,
                emb_dropout=0.2,
                freeze_embeddings=True
            ),
            train_loader, dev_loader,
            epochs=100,
            lr=0.0001
        )
        print(f"Final Dev Accuracy: {glove_2_dev[-1]:.4f}")

        # Plot results
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(glove_2_train, label='2-Layer Train', linewidth=2)
        plt.plot(glove_2_dev, label='2-Layer Dev', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'2-Layer GloVe (Best: {glove_2_dev[-1]:.4f})')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig('dan_glove_results.png', dpi=100)
        print("\n✓ Saved: dan_glove_results.png")
        
        # ========== PART 1B: Random Embeddings ==========
        print("\n" + "=" * 60)
        print("PART 1B: DAN with Random Embeddings (Trainable)")
        print("=" * 60)

        # Train final 2-layer Random model with best config (using experiment wrapper)
        print("\nTraining final 2-Layer Random model with best hyperparameters...")
        rand_2_train, rand_2_dev = experiment(
            DAN_2Layer(
                vocab_size=vocab_size,
                embedding_dim=100,
                hidden_size=150,
                dropout=0.3,
                emb_dropout=0.2,
                freeze_embeddings=False
            ),
            train_loader, dev_loader,
            epochs=100,
            lr=0.0001
        )
        print(f"Final Dev Accuracy: {rand_2_dev[-1]:.4f}")
        
        # Plot Random results
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(rand_2_train, label='2-Layer Train')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('DAN Random - Training Accuracy')
        plt.legend()
        plt.grid()
        
        plt.subplot(1, 2, 2)
        plt.plot(rand_2_dev, label='2-Layer Dev')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('DAN Random - Dev Accuracy')
        plt.legend()
        plt.grid()
        
        plt.tight_layout()
        plt.savefig('dan_random_results.png', dpi=100)
        print("✓ Saved: dan_random_results.png")

        
# ==================== SUBWORDDAN (BPE Only) ====================
    elif args.model == "SUBWORDDAN":
        print("=" * 60)
        print("TRAINING SUBWORD DAN (BPE) MODELS")
        print("=" * 60)

        # Learn BPE merges from train.txt and segment on-the-fly
        from bpe_utils import BPE, BPESentimentDataset
        print("\nLearning BPE merges from train.txt...")
        with open('data/train.txt', 'r', encoding='utf-8') as f:
            corpus = [line.strip().split('\t', 1)[-1] for line in f if line.strip()]
        bpe = BPE(num_merges=10000)  # Change num_merges to experiment with vocab size
        bpe.fit(corpus)

        print("\nLoading BPE-segmented data (in memory)...")
        train_data = BPESentimentDataset('data/train.txt', bpe)
        dev_data = BPESentimentDataset('data/dev.txt', bpe, word_to_idx=train_data.word_to_idx, train=False)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        dev_loader = DataLoader(dev_data, batch_size=16, shuffle=False)
        vocab_size = len(train_data.word_to_idx)
        print(f"BPE Vocabulary size: {vocab_size}")

        # Train final 2-layer BPE model with best config (using experiment wrapper)
        print("\nTraining final 2-Layer BPE model with best hyperparameters...")
        bpe_2_train, bpe_2_dev = experiment(
            DAN_2Layer(
                vocab_size=vocab_size,
                embedding_dim=100,
                hidden_size=150,
                dropout=0.3,
                emb_dropout=0.2,
                freeze_embeddings=False
            ),
            train_loader, dev_loader,
            epochs=100,
            lr=0.0001
        )
        print(f"Final Dev Accuracy: {bpe_2_dev[-1]:.4f}")

        # Plot BPE results
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(bpe_2_train, label='2-Layer Train')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('DAN BPE - Training Accuracy')
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(bpe_2_dev, label='2-Layer Dev')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('DAN BPE - Dev Accuracy')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig('dan_bpe_results.png', dpi=100)
        print("✓ Saved: dan_bpe_results.png")



if __name__ == "__main__":
    main()