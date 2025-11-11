"""
Module d'entraînement du modèle de classification des pièces d'échecs.
"""
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
import cv2
import json
import os
import sys
# Ajoute le répertoire src au path pour les imports relatifs
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from constant import (
    NUM_EPOCHS, BATCH_SIZE, CNN_DATA_FOLDER, 
    IMAGE_SIZE, PIECE_CLASSES, MODEL_WEIGHTS_FILE, CLASS_INDICES_FILE
)


def yoco_create_data_generators(data_folder_path=CNN_DATA_FOLDER):
    """
    Crée des générateurs de données pour l'entraînement et la validation.
    Les générateurs chargent les images au fur et à mesure, ce qui est utile
    pour travailler avec de gros ensembles de données qui ne peuvent pas être
    chargés entièrement en mémoire.
    
    Args:
        data_folder_path: Dossier contenant les sous-dossiers train et validation
        
    Returns:
        Tuple (train_generator, validation_generator)
    """
    # Normalise les images en divisant par 255
    train_data_generator = ImageDataGenerator(rescale=1/255)
    
    # Crée le générateur pour les données d'entraînement
    train_generator_result = train_data_generator.flow_from_directory(
        data_folder_path + 'train',
        target_size=IMAGE_SIZE,  # Redimensionne toutes les images à cette taille
        batch_size=BATCH_SIZE,
        classes=PIECE_CLASSES,  # Spécifie l'ordre des classes
        class_mode='categorical'  # Utilise des labels catégoriels pour la cross-entropy
    )
    
    # Sauvegarde l'ordre des classes pour l'utiliser plus tard
    class_indices_dict = train_generator_result.class_indices
    with open(CLASS_INDICES_FILE, 'w') as f:
        json.dump(class_indices_dict, f)
    
    # Crée le générateur pour les données de validation
    validation_data_generator = ImageDataGenerator(rescale=1/255)
    validation_generator_result = validation_data_generator.flow_from_directory(
        data_folder_path + 'validation',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    return (train_generator_result, validation_generator_result)


def yoco_create_cnn_model(optimizer_instance=RMSprop(learning_rate=0.001)):
    """
    Crée et compile l'architecture du réseau de neurones convolutif.
    
    Args:
        optimizer_instance: Optimiseur à utiliser pour l'entraînement
        
    Returns:
        Modèle compilé et prêt pour l'entraînement
    """
    height, width = IMAGE_SIZE
    num_classes_count = len(PIECE_CLASSES)
    
    cnn_model = Sequential([
        # Première couche de convolution avec 16 filtres
        Conv2D(16, (3, 3), activation='relu', input_shape=(height, width, 3)),
        MaxPooling2D(2, 2),
        
        # Deuxième couche de convolution avec 32 filtres
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Troisième couche de convolution avec 64 filtres
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Quatrième couche de convolution avec 64 filtres
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Cinquième couche de convolution avec 64 filtres
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Aplatit les résultats pour les passer à la couche dense
        Flatten(),
        
        # Couche dense avec 128 neurones
        Dense(128, activation='relu'),
        
        # Couche de sortie avec softmax pour la classification
        Dense(num_classes_count, activation='softmax')
    ])

    cnn_model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer_instance,
                  metrics=['acc'])

    return cnn_model


def yoco_train_cnn_model(cnn_model, train_generator_instance, validation_generator_instance, callbacks_list=[], should_save=False, weights_filename=""):
    """
    Entraîne le modèle avec les données fournies.
    
    Args:
        cnn_model: Modèle à entraîner
        train_generator_instance: Générateur de données d'entraînement
        validation_generator_instance: Générateur de données de validation
        callbacks_list: Liste de callbacks pour sauvegarder des résultats intermédiaires
        should_save: Si True, sauvegarde les poids du modèle
        weights_filename: Nom du fichier pour sauvegarder les poids
        
    Returns:
        Historique de l'entraînement (pour afficher les graphiques)
    """
    total_samples_count = train_generator_instance.n

    training_history = cnn_model.fit(
        train_generator_instance,
        steps_per_epoch=int(total_samples_count / BATCH_SIZE),
        epochs=NUM_EPOCHS,
        verbose=1,
        validation_data=validation_generator_instance,
        callbacks=callbacks_list)

    if should_save:
        cnn_model.save_weights(weights_filename)

    return training_history


def yoco_plot_training_accuracy(training_history):
    """Affiche un graphique de la précision du modèle au cours de l'entraînement."""
    plt.figure(figsize=(7, 4))
    plt.plot([i+1 for i in range(NUM_EPOCHS)],
             training_history.history['acc'], '-o', c='k', lw=2, markersize=9)
    plt.grid(True)
    plt.title("Précision d'entraînement par époque\n", fontsize=18)
    plt.xlabel("Époques d'entraînement", fontsize=15)
    plt.ylabel("Précision d'entraînement", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()


def yoco_plot_training_loss(training_history):
    """Affiche un graphique de la perte du modèle au cours de l'entraînement."""
    plt.figure(figsize=(7, 4))
    plt.plot([i+1 for i in range(NUM_EPOCHS)],
             training_history.history['loss'], '-o', c='k', lw=2, markersize=9)
    plt.grid(True)
    plt.title("Perte d'entraînement par époque\n", fontsize=18)
    plt.xlabel("Époques d'entraînement", fontsize=15)
    plt.ylabel("Perte d'entraînement", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()


def yoco_save_training_history(training_history, history_filename="./history.json"):
    """Sauvegarde l'historique d'entraînement dans un fichier JSON."""
    history_dictionary = training_history.history
    json.dump(history_dictionary, open(history_filename, 'w'))


def yoco_load_training_history(history_filename="./history.json"):
    """Charge l'historique d'entraînement depuis un fichier JSON."""
    with open(history_filename) as json_file:
        history_data = json.load(json_file)
    return history_data


def yoco_test_cnn_model(cnn_model):
    """
    Teste le modèle sur l'ensemble de test et affiche sa précision.
    
    Args:
        cnn_model: Modèle à tester
    """
    test_data_folder = CNN_DATA_FOLDER + 'test'
    height, width = IMAGE_SIZE
    
    correct_predictions_count = 0
    total_samples_count = 0
    
    for subdirectory, subdirs, file_list in os.walk(test_data_folder):
        for filename_item in file_list:
            if filename_item == ".DS_Store":
                continue
            
            piece_class_name = subdirectory.split('/')[-1]
            image_file_path = os.path.join(subdirectory, filename_item)
            
            # Charge et normalise l'image
            image_array = cv2.imread(image_file_path).reshape(1, height, width, 3) / 255.0
            
            # Prédit la classe
            predictions_array = cnn_model.predict(image_array, verbose=0)
            predicted_class_index = predictions_array.argmax()
            
            # Vérifie si la prédiction est correcte
            if predicted_class_index < len(PIECE_CLASSES):
                predicted_class_name = PIECE_CLASSES[predicted_class_index]
                if piece_class_name == predicted_class_name:
                    correct_predictions_count += 1
            total_samples_count += 1
    
    accuracy_score = correct_predictions_count / total_samples_count if total_samples_count > 0 else 0
    print(f"PRÉCISION SUR L'ENSEMBLE DE TEST: {accuracy_score:.4f}")


if __name__ == '__main__':
    train_gen, validation_gen = yoco_create_data_generators(CNN_DATA_FOLDER)
    model = yoco_create_cnn_model()
    history = yoco_train_cnn_model(model, train_gen,
                        validation_gen, should_save=False)
    yoco_save_training_history(history, "./history.json")
    # yoco_plot_training_accuracy(history)
    # yoco_plot_training_loss(history)
    yoco_test_cnn_model(model)
    model.save_weights(MODEL_WEIGHTS_FILE)

