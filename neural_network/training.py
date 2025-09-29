import torch
from torch.utils.data import DataLoader
from crnn import create_model
from ctc_loss import CTCLoss, create_vocabulary
from data_loader import create_data_loaders

# Vérification GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device}')

def train_model():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Charger les données
    data_dir = "/Users/nath/Desktop/OCR_lifprojet/data"
    train_loader, val_loader, char_to_int, int_to_char, vocab_size = create_data_loaders(data_dir)
    
    # Modèle
    model = create_model(vocab_size).to(device)
    criterion = CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    num_epochs = 100
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # =============== TRAINING ===============
        model.train()
        train_loss = 0.0
        num_batches = 0
        
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
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / num_batches
        
        # =============== VALIDATION ===============
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
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
        
        avg_val_loss = val_loss / num_val_batches
        
        print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Scheduler
        scheduler.step(avg_val_loss)
        
        # Sauvegarde du meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_crnn_model.pth')
            print(f'Nouveau meilleur modèle sauvegardé!')

# Lancement de l'entraînement
train_model()
