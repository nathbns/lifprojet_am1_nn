import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class HandwritingDataset(Dataset):
    def __init__(self, image_paths, labels, char_to_int, transform=None, 
                 img_height=32, img_width=128):
        self.image_paths = image_paths
        self.labels = labels
        self.char_to_int = char_to_int
        self.transform = transform
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


# Fonction de collate personnalisée
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