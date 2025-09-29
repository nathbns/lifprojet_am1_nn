import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import re

class IAMDataset(Dataset):
    def __init__(self, image_paths, labels, char_to_int, img_height=32, img_width=128):
        self.image_paths = image_paths
        self.labels = labels
        self.char_to_int = char_to_int
        self.img_height = img_height
        self.img_width = img_width
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Chargement image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            # Image de remplacement si erreur
            image = np.ones((self.img_height, self.img_width), dtype=np.uint8) * 255
        
        # Redimensionnement avec préservation du ratio
        image = self.resize_image(image)
        
        # Normalisation
        image = image.astype(np.float32) / 255.0
        
        # Conversion en tensor
        image = torch.FloatTensor(image).unsqueeze(0)  # [1, H, W]
        
        # Encodage du label
        label = self.labels[idx]
        encoded_label = [self.char_to_int.get(c, 0) for c in label]
        encoded_label = torch.LongTensor(encoded_label)
        
        return image, encoded_label, len(encoded_label)
    
    def resize_image(self, image):
        h, w = image.shape
        
        # Calcul nouveau ratio
        ratio = self.img_width / w
        new_height = int(h * ratio)
        
        if new_height > self.img_height:
            ratio = self.img_height / h
            new_width = int(w * ratio)
            new_height = self.img_height
        else:
            new_width = self.img_width
        
        # Redimensionnement
        image = cv2.resize(image, (new_width, new_height))
        
        # Padding si nécessaire
        if new_height < self.img_height:
            padding = self.img_height - new_height
            image = cv2.copyMakeBorder(image, 0, padding, 0, 0, 
                                     cv2.BORDER_CONSTANT, value=255)
        
        if new_width < self.img_width:
            padding = self.img_width - new_width
            image = cv2.copyMakeBorder(image, 0, 0, 0, padding, 
                                     cv2.BORDER_CONSTANT, value=255)
        
        return image

def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Padding des labels
    max_label_len = max(label_lengths)
    padded_labels = []
    
    for label in labels:
        padded = torch.zeros(max_label_len, dtype=torch.long)
        padded[:len(label)] = label
        padded_labels.append(padded)
    
    padded_labels = torch.stack(padded_labels, 0)
    label_lengths = torch.LongTensor(label_lengths)
    
    return images, padded_labels, label_lengths

def create_vocabulary():
    """Créer le vocabulaire à partir des caractères présents dans les données"""
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?-'
    char_to_int = {char: i+1 for i, char in enumerate(chars)}
    char_to_int['<BLANK>'] = 0  # Token blank pour CTC
    int_to_char = {v: k for k, v in char_to_int.items()}
    int_to_char[0] = ''  # Blank token
    
    return char_to_int, int_to_char, len(char_to_int)

def load_iam_data(data_dir):
    """
    Charger les données IAM depuis le dossier data/
    
    Args:
        data_dir: Chemin vers le dossier data/
    
    Returns:
        image_paths: Liste des chemins vers les images
        labels: Liste des transcriptions correspondantes
    """
    words_file = os.path.join(data_dir, 'iam_words', 'words.txt')
    words_dir = os.path.join(data_dir, 'iam_words', 'words')
    
    image_paths = []
    labels = []
    
    print("Chargement des données IAM...")
    
    with open(words_file, 'r') as f:
        lines = f.readlines()
    
    valid_count = 0
    total_count = 0
    
    for line in lines:
        # Ignorer les commentaires
        if line.startswith('#'):
            continue
            
        parts = line.strip().split()
        if len(parts) < 9:
            continue
            
        word_id = parts[0]
        status = parts[1]
        transcription = parts[-1]  # Dernière partie est la transcription
        
        # Ne prendre que les mots avec segmentation correcte
        if status != 'ok':
            continue
            
        # Construire le chemin de l'image
        # Format: a01-000u-00-00 -> a01/a01-000u/a01-000u-00-00.png
        parts = word_id.split('-')
        form_id = parts[0]  # a01
        line_id = f"{parts[0]}-{parts[1]}"  # a01-000u
        image_path = os.path.join(words_dir, form_id, line_id, f"{word_id}.png")
        
        # Vérifier que l'image existe
        if os.path.exists(image_path):
            # Nettoyer la transcription (enlever les caractères spéciaux)
            clean_transcription = re.sub(r'[^a-zA-Z0-9\s.,;:!?-]', '', transcription)
            if len(clean_transcription) > 0:
                image_paths.append(image_path)
                labels.append(clean_transcription)
                valid_count += 1
        
        total_count += 1
        
        if total_count % 10000 == 0:
            print(f"Traité {total_count} lignes, {valid_count} images valides trouvées")
    
    print(f"Chargement terminé: {valid_count} images valides sur {total_count} lignes")
    return image_paths, labels

def create_data_loaders(data_dir, batch_size=32, test_size=0.2, random_state=42):
    """
    Créer les DataLoaders pour l'entraînement et la validation
    
    Args:
        data_dir: Chemin vers le dossier data/
        batch_size: Taille des batches
        test_size: Proportion des données pour la validation
        random_state: Seed pour la reproductibilité
    
    Returns:
        train_loader, val_loader, char_to_int, int_to_char, vocab_size
    """
    # Charger les données
    image_paths, labels = load_iam_data(data_dir)
    
    # Créer le vocabulaire
    char_to_int, int_to_char, vocab_size = create_vocabulary()
    
    # Diviser en train/validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state
    )
    
    print(f"Données d'entraînement: {len(train_paths)}")
    print(f"Données de validation: {len(val_paths)}")
    print(f"Taille du vocabulaire: {vocab_size}")
    
    # Créer les datasets
    train_dataset = IAMDataset(train_paths, train_labels, char_to_int)
    val_dataset = IAMDataset(val_paths, val_labels, char_to_int)
    
    # Créer les DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=4
    )
    
    return train_loader, val_loader, char_to_int, int_to_char, vocab_size

if __name__ == "__main__":
    # Test du chargement des données
    data_dir = "OCR_lifprojet/data"
    
    print("Test du chargement des données...")
    train_loader, val_loader, char_to_int, int_to_char, vocab_size = create_data_loaders(
        data_dir, batch_size=8, test_size=0.1
    )
    
    print(f"Vocabulaire créé avec {vocab_size} caractères")
    print(f"Exemple de mapping: 'a' -> {char_to_int.get('a', 'N/A')}")
    print(f"Exemple de mapping inverse: {char_to_int.get('a', 0)} -> '{int_to_char.get(char_to_int.get('a', 0), 'N/A')}'")
    
    # Test d'un batch
    print("\nTest d'un batch d'entraînement...")
    for images, targets, target_lengths in train_loader:
        print(f"Images shape: {images.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Target lengths: {target_lengths}")
        print(f"Premier label: {targets[0][:target_lengths[0]]}")
        break
