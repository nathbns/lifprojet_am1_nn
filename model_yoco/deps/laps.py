"""
Modèle CNN pour la classification des intersections d'échiquier.

Ce module définit l'architecture du réseau de neurones utilisé
pour valider si une région d'image correspond à une intersection
valide des lignes de l'échiquier.

Architecture du modèle :
- Entrée: Image 21x21 pixels en niveaux de gris
- 2 blocs convolutionnels avec BatchNormalization
- Couche dense avec Dropout
- Sortie: 2 classes (valide / invalide)

Note: Ce modèle a été entraîné séparément et les poids
sont stockés dans data/laps_models/
"""

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPooling2D, 
    BatchNormalization, Dropout, Flatten
)

# Constantes du modèle
INPUT_SHAPE = (21, 21, 1)  # Image 21x21 en niveaux de gris
NUM_CLASSES = 2            # Valide ou Invalide
LEARNING_RATE = 0.001


def create_laps_model() -> Sequential:
    """
    Crée le modèle CNN pour la classification des intersections.
    
    L'architecture utilise plusieurs couches de convolution avec
    des tailles de kernel décroissantes (3, 2, 1) pour capturer
    des features à différentes échelles.
    
    Returns:
        Modèle Keras compilé
    """
    model = Sequential(name="LAPS_Intersection_Classifier")
    
    # Couche d'entrée
    model.add(Dense(
        INPUT_SHAPE[0] * INPUT_SHAPE[1],
        input_shape=INPUT_SHAPE,
        name="input_dense"
    ))
    
    # Blocs convolutionnels (répétés 2 fois)
    for block_idx in range(2):
        # Convolutions avec kernels de tailles décroissantes
        for kernel_size in [3, 2, 1]:
            model.add(Conv2D(
                filters=16,
                kernel_size=kernel_size,
                activation='elu',
                name=f"conv_block{block_idx+1}_k{kernel_size}"
            ))
        
        # Pooling et normalisation
        model.add(MaxPooling2D(
            pool_size=(2, 2),
            name=f"pool_block{block_idx+1}"
        ))
        model.add(BatchNormalization(
            name=f"batchnorm_block{block_idx+1}"
        ))
    
    # Couches denses
    model.add(Dense(
        128,
        activation='elu',
        name="dense_features"
    ))
    model.add(Dropout(0.5, name="dropout"))
    model.add(Flatten(name="flatten"))
    
    # Couche de sortie
    model.add(Dense(
        NUM_CLASSES,
        activation='softmax',
        name="output"
    ))
    
    # Compilation
    model.compile(
        optimizer=RMSprop(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    return model


# Crée une instance du modèle pour l'import
model = create_laps_model()


def get_model_summary():
    """Affiche un résumé de l'architecture du modèle."""
    model.summary()


def predict_intersection(image_patch):
    """
    Prédit si une région d'image est une intersection valide.
    
    Args:
        image_patch: Image 21x21 normalisée (valeurs 0 ou 1)
                    de shape (batch, 21, 21, 1)
                    
    Returns:
        Probabilités [prob_valide, prob_invalide]
    """
    return model.predict(image_patch, verbose=0)
