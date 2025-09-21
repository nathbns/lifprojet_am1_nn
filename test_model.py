import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from crnn import create_model
from ctc_loss import create_vocabulary
from decode_ctc import ctc_decode_greedy

def load_model(model_path, device=None):
    """Charger un modèle entraîné"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = create_model(checkpoint['vocab_size']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['int_to_char'], checkpoint['char_to_int']

def preprocess_image(image_path, img_height=32, img_width=128):
    """Préprocesser une image pour le modèle"""
    # Charger l'image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")
    
    # Redimensionnement avec préservation du ratio
    h, w = image.shape
    ratio = img_width / w
    new_height = int(h * ratio)
    
    if new_height > img_height:
        ratio = img_height / h
        new_width = int(w * ratio)
        new_height = img_height
    else:
        new_width = img_width
    
    # Redimensionnement
    image = cv2.resize(image, (new_width, new_height))
    
    # Padding si nécessaire
    if new_height < img_height:
        padding = img_height - new_height
        image = cv2.copyMakeBorder(image, 0, padding, 0, 0, 
                                 cv2.BORDER_CONSTANT, value=255)
    
    if new_width < img_width:
        padding = img_width - new_width
        image = cv2.copyMakeBorder(image, 0, 0, 0, padding, 
                                 cv2.BORDER_CONSTANT, value=255)
    
    # Normalisation
    image = image.astype(np.float32) / 255.0
    
    return image

def predict_single_image(model, image_path, int_to_char, device=None):
    """Prédire le texte d'une image"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Préprocesser l'image
    image = preprocess_image(image_path)
    
    # Convertir en tensor
    image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(device)
    
    # Prédiction
    with torch.no_grad():
        log_probs = model(image_tensor)
        predicted_texts = ctc_decode_greedy(log_probs, int_to_char)
    
    return predicted_texts[0]

def test_on_sample_images(data_dir, model_path, num_samples=10):
    """Tester le modèle sur quelques images d'échantillon"""
    print(f"Test du modèle sur {num_samples} images...")
    
    # Charger le modèle
    model, int_to_char, char_to_int = load_model(model_path)
    device = next(model.parameters()).device
    
    # Trouver quelques images de test
    words_dir = os.path.join(data_dir, 'iam_words', 'words')
    sample_images = []
    
    # Parcourir quelques dossiers pour trouver des images
    for form_id in ['a01', 'a02', 'a03'][:3]:  # Limiter à 3 formes
        form_dir = os.path.join(words_dir, form_id)
        if os.path.exists(form_dir):
            for line_dir in os.listdir(form_dir)[:2]:  # 2 lignes par forme
                line_path = os.path.join(form_dir, line_dir)
                if os.path.isdir(line_path):
                    for img_file in os.listdir(line_path)[:2]:  # 2 images par ligne
                        if img_file.endswith('.png'):
                            sample_images.append(os.path.join(line_path, img_file))
                            if len(sample_images) >= num_samples:
                                break
                if len(sample_images) >= num_samples:
                    break
        if len(sample_images) >= num_samples:
            break
    
    print(f"Trouvé {len(sample_images)} images de test")
    
    # Tester chaque image
    results = []
    for i, image_path in enumerate(sample_images[:num_samples]):
        try:
            # Obtenir le vrai label depuis le nom du fichier
            filename = os.path.basename(image_path)
            word_id = filename.replace('.png', '')
            
            # Prédire
            predicted_text = predict_single_image(model, image_path, int_to_char, device)
            
            print(f"\nImage {i+1}: {filename}")
            print(f"  Prédit: '{predicted_text}'")
            
            # Afficher l'image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            plt.figure(figsize=(8, 2))
            plt.imshow(image, cmap='gray')
            plt.title(f"Prédit: '{predicted_text}'")
            plt.axis('off')
            plt.show()
            
            results.append({
                'image_path': image_path,
                'predicted': predicted_text,
                'word_id': word_id
            })
            
        except Exception as e:
            print(f"Erreur avec l'image {image_path}: {e}")
    
    return results

def interactive_test(model_path):
    """Test interactif - l'utilisateur peut entrer le chemin d'une image"""
    print("=== TEST INTERACTIF ===")
    print("Entrez le chemin vers une image PNG pour la tester")
    print("(ou 'quit' pour quitter)")
    
    # Charger le modèle
    model, int_to_char, char_to_int = load_model(model_path)
    device = next(model.parameters()).device
    
    while True:
        try:
            image_path = input("\nChemin de l'image: ").strip()
            
            if image_path.lower() == 'quit':
                break
            
            if not os.path.exists(image_path):
                print("Fichier non trouvé!")
                continue
            
            if not image_path.lower().endswith('.png'):
                print("Veuillez fournir un fichier PNG")
                continue
            
            # Prédire
            predicted_text = predict_single_image(model, image_path, int_to_char, device)
            
            print(f"Texte prédit: '{predicted_text}'")
            
            # Afficher l'image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            plt.figure(figsize=(10, 3))
            plt.imshow(image, cmap='gray')
            plt.title(f"Prédit: '{predicted_text}'")
            plt.axis('off')
            plt.show()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Erreur: {e}")

if __name__ == "__main__":
    data_dir = "OCR_lifprojet/data"
    
    print("=== TEST DU MODÈLE CRNN ===")
    print("Choisissez le mode de test:")
    print("1. Test sur échantillons automatiques")
    print("2. Test interactif (entrer chemin d'image)")
    
    try:
        choice = input("Votre choix (1-2): ").strip()
        
        # Chercher le meilleur modèle disponible
        model_files = [
            'models/best_crnn_phase_3.pth',
            'models/best_crnn_phase_2.pth', 
            'models/best_crnn_phase_1.pth',
            'models/best_crnn_small.pth'
        ]
        
        model_path = None
        for model_file in model_files:
            if os.path.exists(model_file):
                model_path = model_file
                break
        
        if model_path is None:
            print("Aucun modèle entraîné trouvé!")
            print("Lancez d'abord l'entraînement avec: python3 train_progressive.py")
            exit(1)
        
        print(f"Utilisation du modèle: {model_path}")
        
        if choice == "1":
            test_on_sample_images(data_dir, model_path, num_samples=5)
        elif choice == "2":
            interactive_test(model_path)
        else:
            print("Choix invalide. Lancement du test automatique.")
            test_on_sample_images(data_dir, model_path, num_samples=5)
            
    except KeyboardInterrupt:
        print("\nTest interrompu.")
    except Exception as e:
        print(f"Erreur: {e}")

