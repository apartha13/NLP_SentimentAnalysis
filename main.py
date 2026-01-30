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

def hyperparameter_search_random(vocab_size, train_loader, dev_loader, 
                                  model_class, embedding_dim=100):
    """
    Search through hyperparameters for Random Embedding models
    """
    hidden_sizes = [150,200,250]
    learning_rates = [0.0001, 0.0005, 0.001]
    dropouts = [0.3, 0.5, 0.7]
    emb_dropouts = [0.0, 0.2, 0.5]

    results = []
    best_accuracy = 0
    best_config = None

    total_configs = len(hidden_sizes) * len(learning_rates) * len(dropouts) * len(emb_dropouts)
    config_num = 0

    print(f"\nSearching {total_configs} configurations...")
    print("=" * 80)

    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            for dropout in dropouts:
                for emb_dropout in emb_dropouts:
                    config_num += 1
                    # Create model with emb_dropout if supported
                    try:
                        model = model_class(
                            vocab_size=vocab_size,
                            embedding_dim=embedding_dim,
                            hidden_size=hidden_size,
                            dropout=dropout,
                            emb_dropout=emb_dropout,
                            freeze_embeddings=False
                        )
                    except TypeError:
                        model = model_class(
                            vocab_size=vocab_size,
                            embedding_dim=embedding_dim,
                            hidden_size=hidden_size,
                            dropout=dropout,
                            freeze_embeddings=False
                        )


                    # Train using experiment (returns train_acc, dev_acc)
                    train_acc, dev_acc = experiment(
                        model, train_loader, dev_loader,
                        epochs=100, lr=lr
                    )

                    final_dev_acc = dev_acc[-1]
                    final_train_acc = train_acc[-1]

                    # Store result
                    results.append({
                        'hidden_size': hidden_size,
                        'learning_rate': lr,
                        'dropout': dropout,
                        'emb_dropout': emb_dropout,
                        'final_train_acc': final_train_acc,
                        'final_dev_acc': final_dev_acc,
                        'train_history': train_acc,
                        'dev_history': dev_acc
                    })

                    # Update best
                    if final_dev_acc > best_accuracy:
                        best_accuracy = final_dev_acc
                        best_config = {
                            'hidden_size': hidden_size,
                            'learning_rate': lr,
                            'dropout': dropout,
                            'emb_dropout': emb_dropout
                        }
                        print(f"[{config_num}/{total_configs}] ✓ NEW BEST: {best_accuracy:.4f} "
                              f"| Hidden={hidden_size} | LR={lr} | Dropout={dropout} | EmbDropout={emb_dropout}")
                    else:
                        if config_num % 10 == 0:
                            print(f"[{config_num}/{total_configs}] Tested: Dev Acc={final_dev_acc:.4f}")

    print("=" * 80)
    print(f"\nBEST CONFIGURATION FOUND:")
    print(f"  Hidden Size: {best_config['hidden_size']}")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Dropout: {best_config['dropout']}")
    print(f"  Embedding Dropout: {best_config['emb_dropout']}")
    print(f"  Dev Accuracy: {best_accuracy:.4f}")

    # Print top 10 results
    print("\nTop 10 Configurations:")
    print(f"{'Rank':<6} {'Hidden':<10} {'LR':<10} {'Dropout':<10} {'EmbDrop':<10} {'Train Acc':<12} {'Dev Acc':<12}")
    print("-" * 80)

    sorted_results = sorted(results, key=lambda x: x['final_dev_acc'], reverse=True)
    for i, result in enumerate(sorted_results[:10], 1):
        print(f"{i:<6} {result['hidden_size']:<10} {result['learning_rate']:<10} "
              f"{result['dropout']:<10} {result['emb_dropout']:<10} {result['final_train_acc']:<12.4f} {result['final_dev_acc']:<12.4f}")

    return best_config, results

def hyperparameter_search_glove(pretrained_embeddings, train_loader, dev_loader, 
                                 model_class, num_layers):
    """
    Search through hyperparameters for GloVe models
    """
    hidden_sizes = [150, 250]
    learning_rates = [0.0001]
    dropouts = [0.05, 0.1, 0.15]
    emb_dropouts = [0.0]

    results = []
    best_accuracy = 0
    best_config = None

    total_configs = len(hidden_sizes) * len(learning_rates) * len(dropouts) * len(emb_dropouts)
    config_num = 0

    print(f"\nSearching {total_configs} configurations...")
    print("=" * 80)

    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            for dropout in dropouts:
                for emb_dropout in emb_dropouts:
                    config_num += 1
                    # Create model with emb_dropout if supported
                    try:
                        model = model_class(
                            pretrained_embeddings=pretrained_embeddings,
                            hidden_size=hidden_size,
                            dropout=dropout,
                            emb_dropout=emb_dropout,
                            freeze_embeddings=True
                        )
                    except TypeError:
                        model = model_class(
                            pretrained_embeddings=pretrained_embeddings,
                            hidden_size=hidden_size,
                            dropout=dropout,
                            freeze_embeddings=True
                        )

                    # Train using experiment (returns train_acc, dev_acc)
                    train_acc, dev_acc = experiment(
                        model, train_loader, dev_loader,
                        epochs=100, lr=lr
                    )

                    final_dev_acc = dev_acc[-1]
                    final_train_acc = train_acc[-1]

                    # Store result
                    results.append({
                        'hidden_size': hidden_size,
                        'learning_rate': lr,
                        'dropout': dropout,
                        'emb_dropout': emb_dropout,
                        'final_train_acc': final_train_acc,
                        'final_dev_acc': final_dev_acc,
                        'train_history': train_acc,
                        'dev_history': dev_acc
                    })

                    # Update best
                    if final_dev_acc > best_accuracy:
                        best_accuracy = final_dev_acc
                        best_config = {
                            'hidden_size': hidden_size,
                            'learning_rate': lr,
                            'dropout': dropout,
                            'emb_dropout': emb_dropout
                        }
                        print(f"[{config_num}/{total_configs}] ✓ NEW BEST: {best_accuracy:.4f} "
                              f"| Hidden={hidden_size} | LR={lr} | Dropout={dropout} | EmbDropout={emb_dropout}")
                    else:
                        if config_num % 10 == 0:
                            print(f"[{config_num}/{total_configs}] Tested: Dev Acc={final_dev_acc:.4f}")

    print("=" * 80)
    print(f"\nBEST CONFIGURATION FOUND:")
    print(f"  Hidden Size: {best_config['hidden_size']}")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Dropout: {best_config['dropout']}")
    print(f"  Embedding Dropout: {best_config['emb_dropout']}")
    print(f"  Dev Accuracy: {best_accuracy:.4f}")

    # Print top 10 results
    print("\nTop 10 Configurations:")
    print(f"{'Rank':<6} {'Hidden':<10} {'LR':<10} {'Dropout':<10} {'EmbDrop':<10} {'Train Acc':<12} {'Dev Acc':<12}")
    print("-" * 80)

    sorted_results = sorted(results, key=lambda x: x['final_dev_acc'], reverse=True)
    for i, result in enumerate(sorted_results[:10], 1):
        print(f"{i:<6} {result['hidden_size']:<10} {result['learning_rate']:<10} "
              f"{result['dropout']:<10} {result['emb_dropout']:<10} {result['final_train_acc']:<12.4f} {result['final_dev_acc']:<12.4f}")

    return best_config, results

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


def train_model(model, train_loader, dev_loader, epochs=100, learning_rate=0.001,
                weight_decay=0.0, save_path=None):
    """Train a model and return accuracy histories"""
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    train_accs = []
    dev_accs = []
    best_dev = float('-inf')
    
    for epoch in range(epochs):
        train_acc, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        dev_acc, dev_loss = eval_epoch(dev_loader, model, loss_fn, optimizer)
        
        train_accs.append(train_acc)
        dev_accs.append(dev_acc)
        
        # Save best checkpoint (no early stopping)
        if dev_acc > best_dev:
            best_dev = dev_acc
            if save_path:
                torch.save(model.state_dict(), save_path)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs} | Train Acc: {train_acc:.4f} | Dev Acc: {dev_acc:.4f}')
    
    # make the last reported dev acc equal to the best seen (so callers that use dev_accs[-1] get the actual best)
    if dev_accs:
        dev_accs[-1] = best_dev
    
    return train_accs, dev_accs


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

        # Hyperparameter search for DAN_2Layer (BPE)
        """print("\nDAN BPE - Hyperparameter Search:")
        best_config_bpe, bpe_results = hyperparameter_search_random(
            vocab_size, train_loader, dev_loader,
            DAN_2Layer, embedding_dim=100
        )
        print("Best config for DAN_2Layer (BPE):", best_config_bpe)"""

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