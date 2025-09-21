import torch
import cv2
import numpy as np

def ctc_decode_greedy(log_probs, int_to_char, blank=0):
    """
    Décodage CTC greedy (plus probable à chaque étape)
    """
    # Obtenir les indices les plus probables
    _, max_indices = torch.max(log_probs, dim=2)  # [seq_len, batch]
    
    decoded_texts = []
    
    for batch_idx in range(max_indices.size(1)):
        sequence = max_indices[:, batch_idx]
        
        # Suppression des duplicatas et blanks
        decoded_chars = []
        prev_char = None
        
        for char_idx in sequence:
            char_idx = char_idx.item()
            
            # Skip blank tokens
            if char_idx == blank:
                prev_char = None
                continue
                
            # Skip duplicates
            if char_idx != prev_char:
                if char_idx in int_to_char:
                    decoded_chars.append(int_to_char[char_idx])
                prev_char = char_idx
        
        decoded_texts.append(''.join(decoded_chars))
    
    return decoded_texts

def predict_text(model, image_path, char_to_int, int_to_char):
    """
    Prédiction sur une seule image
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Chargement et préprocessing
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Redimensionnement (même logique que le dataset)
    h, w = image.shape
    ratio = 128 / w
    new_height = int(h * ratio)
    
    if new_height > 32:
        ratio = 32 / h
        new_width = int(w * ratio)
        new_height = 32
    else:
        new_width = 128
    
    image = cv2.resize(image, (new_width, new_height))
    
    # Padding
    if new_height < 32:
        padding = 32 - new_height
        image = cv2.copyMakeBorder(image, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=255)
    
    if new_width < 128:
        padding = 128 - new_width
        image = cv2.copyMakeBorder(image, 0, 0, 0, padding, cv2.BORDER_CONSTANT, value=255)
    
    # Normalisation et conversion
    image = image.astype(np.float32) / 255.0
    image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 32, 128]
    
    # Prédiction
    with torch.no_grad():
        log_probs = model(image)  # [seq_len, 1, num_classes]
        
    # Décodage
    predicted_text = ctc_decode_greedy(log_probs, int_to_char)[0]
    
    return predicted_text

# Exemple d'utilisation
# model = create_model(vocab_size)
# model.load_state_dict(torch.load('best_crnn_model.pth'))
# text = predict_text(model, 'image.jpg', char_to_int, int_to_char)
# print(f'Texte prédit: {text}')
