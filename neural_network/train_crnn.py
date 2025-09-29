import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from datetime import datetime

# Import de nos modules
from crnn import create_model
from ctc_loss import CTCLoss
from data_loader import create_data_loaders
from decode_ctc import ctc_decode_greedy

def train_model(data_dir, batch_size=32, num_epochs=50, learning_rate=0.001, 
                save_dir='models', device=None):
    """
    Fonction d'entraînement complète pour le modèle CRNN
    
    Args:
        data_dir: Chemin vers le dossier data/
        batch_size: Taille des batches
        num_epochs: Nombre d'époques
        learning_rate: Taux d'apprentissage
        save_dir: Dossier pour sauvegarder les modèles
        device: Device PyTorch (auto-détection si None)
    """
    
    # Configuration du device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device utilisé: {device}")
    
    # Créer le dossier de sauvegarde
    os.makedirs(save_dir, exist_ok=True)
    
    # Charger les données
    print("Chargement des données...")
    train_loader, val_loader, char_to_int, int_to_char, vocab_size = create_data_loaders(
        data_dir, batch_size=batch_size, test_size=0.1
    )
    
    print(f"Vocabulaire: {vocab_size} caractères")
    print(f"Données d'entraînement: {len(train_loader.dataset)}")
    print(f"Données de validation: {len(val_loader.dataset)}")
    
    # Créer le modèle
    model = create_model(vocab_size).to(device)
    print(f"Modèle créé avec {sum(p.numel() for p in model.parameters())} paramètres")
    
    # Fonction de perte et optimiseur
    criterion = CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Variables pour le suivi
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\nDébut de l'entraînement pour {num_epochs} époques...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # =============== PHASE D'ENTRAÎNEMENT ===============
        model.train()
        train_loss = 0.0
        num_train_batches = 0
        
        for batch_idx, (images, targets, target_lengths) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            log_probs = model(images)  # [seq_len, batch, num_classes]
            
            # Longueurs des séquences d'entrée
            input_lengths = torch.full((images.size(0),), log_probs.size(0), 
                                     dtype=torch.long, device=device)
            
            # Calcul de la perte
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping pour éviter l'explosion des gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            num_train_batches += 1
            
            # Affichage du progrès
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / num_train_batches
        train_losses.append(avg_train_loss)
        
        # =============== PHASE DE VALIDATION ===============
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        correct_predictions = 0
        total_predictions = 0
        
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
                num_val_batches += 1
                
                # Test de décodage sur quelques échantillons
                if num_val_batches <= 5:  # Seulement sur les premiers batches
                    predicted_texts = ctc_decode_greedy(log_probs, int_to_char)
                    
                    for i, (pred_text, target_length) in enumerate(zip(predicted_texts, target_lengths)):
                        if i < 3:  # Afficher seulement les 3 premiers
                            target_text = ''.join([int_to_char.get(idx.item(), '') 
                                                 for idx in targets[i][:target_length]])
                            print(f"  Prédit: '{pred_text}' | Réel: '{target_text}'")
                            
                            if pred_text.strip() == target_text.strip():
                                correct_predictions += 1
                            total_predictions += 1
        
        avg_val_loss = val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        # Calcul du temps d'époque
        epoch_time = time.time() - start_time
        
        # Affichage des résultats
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Temps: {epoch_time:.1f}s')
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            print(f'  Précision (échantillon): {accuracy:.2%}')
        
        # Scheduler
        scheduler.step(avg_val_loss)
        
        # Sauvegarde du meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(save_dir, 'best_crnn_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'vocab_size': vocab_size,
                'char_to_int': char_to_int,
                'int_to_char': int_to_char
            }, model_path)
            print(f'  ✓ Nouveau meilleur modèle sauvegardé! (Loss: {avg_val_loss:.4f})')
        
        # Sauvegarde périodique
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'vocab_size': vocab_size,
                'char_to_int': char_to_int,
                'int_to_char': int_to_char
            }, checkpoint_path)
            print(f'  ✓ Checkpoint sauvegardé: {checkpoint_path}')
        
        print("-" * 60)
    
    print(f"\nEntraînement terminé!")
    print(f"Meilleure perte de validation: {best_val_loss:.4f}")
    print(f"Modèles sauvegardés dans: {save_dir}")
    
    return model, train_losses, val_losses

def test_model(model_path, data_dir, device=None):
    """
    Tester le modèle entraîné sur quelques échantillons
    
    Args:
        model_path: Chemin vers le modèle sauvegardé
        data_dir: Chemin vers le dossier data/
        device: Device PyTorch
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Charger le checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Créer le modèle
    model = create_model(checkpoint['vocab_size']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Charger les données de test
    _, val_loader, _, int_to_char, _ = create_data_loaders(data_dir, batch_size=8, test_size=0.1)
    
    print(f"Test du modèle sur {len(val_loader.dataset)} échantillons...")
    print("=" * 50)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, targets, target_lengths) in enumerate(val_loader):
            if batch_idx >= 5:  # Limiter à 5 batches pour le test
                break
                
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            log_probs = model(images)
            predicted_texts = ctc_decode_greedy(log_probs, int_to_char)
            
            for i, (pred_text, target_length) in enumerate(zip(predicted_texts, target_lengths)):
                target_text = ''.join([int_to_char.get(idx.item(), '') 
                                     for idx in targets[i][:target_length]])
                
                print(f"Image {total+1}:")
                print(f"  Prédit: '{pred_text}'")
                print(f"  Réel:   '{target_text}'")
                print(f"  Correct: {pred_text.strip() == target_text.strip()}")
                print()
                
                if pred_text.strip() == target_text.strip():
                    correct += 1
                total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Précision sur {total} échantillons: {accuracy:.2%}")

if __name__ == "__main__":
    # Configuration
    data_dir = "/Users/nath/Desktop/OCR_lifprojet/data"
    batch_size = 16  # Réduire si problème de mémoire
    num_epochs = 20  # Commencer avec moins d'époques pour tester
    learning_rate = 0.001
    
    print("=== ENTRAÎNEMENT DU MODÈLE CRNN ===")
    print(f"Données: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Époques: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print()
    
    # Lancer l'entraînement
    model, train_losses, val_losses = train_model(
        data_dir=data_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_dir='models'
    )
    
    # Test du modèle
    print("\n=== TEST DU MODÈLE ===")
    test_model('models/best_crnn_model.pth', data_dir)
