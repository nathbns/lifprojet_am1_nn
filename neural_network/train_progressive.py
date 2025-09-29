import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime

# Import de nos modules
from crnn import create_model
from ctc_loss import CTCLoss
from data_loader import create_data_loaders
from decode_ctc import ctc_decode_greedy

def train_progressive(data_dir, phase=1, device=None):
    """
    Entraînement progressif en 3 phases
    
    Phase 1: 20 epochs, 1000 échantillons, batch_size=8
    Phase 2: 50 epochs, 5000 échantillons, batch_size=16  
    Phase 3: 100 epochs, 10000 échantillons, batch_size=32
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration selon la phase
    if phase == 1:
        max_samples = 1000
        batch_size = 8
        num_epochs = 20
        learning_rate = 0.001
        print("=== PHASE 1: Test initial (20 epochs) ===")
        print("Objectif: Training loss 16→2, Validation accuracy 0.1→0.5")
        print("Temps estimé: 8-10h")
    elif phase == 2:
        max_samples = 5000
        batch_size = 16
        num_epochs = 50
        learning_rate = 0.0005
        print("=== PHASE 2: Entraînement intermédiaire (50 epochs) ===")
        print("Objectif: Training loss 2→0.5, Validation accuracy 0.5→0.7")
        print("Temps estimé: +20h")
    else:  # phase == 3
        max_samples = 10000
        batch_size = 32
        num_epochs = 100
        learning_rate = 0.0001
        print("=== PHASE 3: Entraînement complet (100 epochs) ===")
        print("Objectif: Training loss 0.5→0.1, Validation accuracy 0.7→0.75")
        print("Temps estimé: +25h")
    
    print(f"Échantillons: {max_samples}, Batch: {batch_size}, LR: {learning_rate}")
    print(f"Device: {device}")
    print()
    
    # Charger les données
    print("Chargement des données...")
    train_loader, val_loader, char_to_int, int_to_char, vocab_size = create_data_loaders(
        data_dir, batch_size=batch_size, test_size=0.2
    )
    
    # Limiter le nombre d'échantillons selon la phase
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    train_indices = list(range(min(max_samples, len(train_dataset))))
    val_indices = list(range(min(max_samples // 4, len(val_dataset))))
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, 
                            collate_fn=train_loader.collate_fn, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, 
                          collate_fn=val_loader.collate_fn, num_workers=0)
    
    print(f"Données d'entraînement: {len(train_subset)}")
    print(f"Données de validation: {len(val_subset)}")
    print(f"Vocabulaire: {vocab_size} caractères")
    
    # Créer ou charger le modèle
    model_path = f'models/crnn_phase_{phase-1}.pth' if phase > 1 else None
    
    if model_path and os.path.exists(model_path):
        print(f"Chargement du modèle de la phase précédente: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model = create_model(checkpoint['vocab_size']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Modèle chargé avec succès!")
    else:
        print("Création d'un nouveau modèle...")
        model = create_model(vocab_size).to(device)
    
    print(f"Modèle: {sum(p.numel() for p in model.parameters())} paramètres")
    
    # Optimiseur et scheduler
    criterion = CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Variables de suivi
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    accuracies = []
    
    print(f"\nDébut de l'entraînement - Phase {phase} ({num_epochs} époques)...")
    print("=" * 80)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # =============== ENTRAÎNEMENT ===============
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, targets, target_lengths) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            
            log_probs = model(images)
            input_lengths = torch.full((images.size(0),), log_probs.size(0), 
                                     dtype=torch.long, device=device)
            
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            # Affichage du progrès
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # =============== VALIDATION ===============
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets, target_lengths in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)
                
                log_probs = model(images)
                input_lengths = torch.full((images.size(0),), log_probs.size(0), 
                                         dtype=torch.long, device=device)
                
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                val_loss += loss.item()
                
                # Test de précision
                predicted_texts = ctc_decode_greedy(log_probs, int_to_char)
                for i, (pred_text, target_length) in enumerate(zip(predicted_texts, target_lengths)):
                    target_text = ''.join([int_to_char.get(idx.item(), '') 
                                         for idx in targets[i][:target_length]])
                    
                    if pred_text.strip() == target_text.strip():
                        correct += 1
                    total += 1
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0
        
        val_losses.append(avg_val_loss)
        accuracies.append(accuracy)
        
        epoch_time = time.time() - epoch_start
        
        # Affichage des résultats
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Accuracy: {accuracy:.2%}, '
              f'Temps: {epoch_time:.1f}s')
        
        # Scheduler
        scheduler.step(avg_val_loss)
        
        # Sauvegarde du meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('models', exist_ok=True)
            
            # Sauvegarde du meilleur modèle de la phase
            best_path = f'models/best_crnn_phase_{phase}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'accuracy': accuracy,
                'vocab_size': vocab_size,
                'char_to_int': char_to_int,
                'int_to_char': int_to_char,
                'phase': phase
            }, best_path)
            
            # Sauvegarde pour la phase suivante
            next_path = f'models/crnn_phase_{phase}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'accuracy': accuracy,
                'vocab_size': vocab_size,
                'char_to_int': char_to_int,
                'int_to_char': int_to_char,
                'phase': phase
            }, next_path)
            
            print(f'  ✓ Meilleur modèle sauvegardé! (Loss: {avg_val_loss:.4f}, Acc: {accuracy:.2%})')
        
        # Sauvegarde périodique
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'models/checkpoint_phase_{phase}_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'accuracies': accuracies,
                'vocab_size': vocab_size,
                'char_to_int': char_to_int,
                'int_to_char': int_to_char,
                'phase': phase
            }, checkpoint_path)
            print(f'  ✓ Checkpoint: {checkpoint_path}')
        
        print("-" * 80)
    
    total_time = time.time() - start_time
    print(f"\nPhase {phase} terminée!")
    print(f"Temps total: {total_time/3600:.1f}h")
    print(f"Meilleure perte: {best_val_loss:.4f}")
    print(f"Meilleure précision: {max(accuracies):.2%}")
    print(f"Modèles sauvegardés dans: models/")
    
    # Graphiques
    plot_training_curves(train_losses, val_losses, accuracies, phase)
    
    return model, train_losses, val_losses, accuracies

def plot_training_curves(train_losses, val_losses, accuracies, phase):
    """Afficher les courbes d'entraînement"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Courbe des pertes
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Phase {phase} - Training Curves')
    ax1.legend()
    ax1.grid(True)
    
    # Courbe de précision
    ax2.plot(accuracies, label='Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Phase {phase} - Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'models/training_curves_phase_{phase}.png')
    plt.show()

if __name__ == "__main__":
    data_dir = "/Users/nath/Desktop/OCR_lifprojet/data"
    
    print("=== ENTRAÎNEMENT PROGRESSIF CRNN ===")
    print("Choisissez la phase à lancer:")
    print("1. Phase 1: Test initial (20 epochs, 1000 échantillons)")
    print("2. Phase 2: Entraînement intermédiaire (50 epochs, 5000 échantillons)")
    print("3. Phase 3: Entraînement complet (100 epochs, 10000 échantillons)")
    print("4. Toutes les phases")
    
    try:
        choice = input("Votre choix (1-4): ").strip()
        
        if choice == "1":
            train_progressive(data_dir, phase=1)
        elif choice == "2":
            train_progressive(data_dir, phase=2)
        elif choice == "3":
            train_progressive(data_dir, phase=3)
        elif choice == "4":
            print("Lancement de toutes les phases...")
            for phase in [1, 2, 3]:
                print(f"\n{'='*60}")
                print(f"PHASE {phase}")
                print(f"{'='*60}")
                train_progressive(data_dir, phase=phase)
        else:
            print("Choix invalide. Lancement de la phase 1 par défaut.")
            train_progressive(data_dir, phase=1)
            
    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur.")
    except Exception as e:
        print(f"Erreur: {e}")


