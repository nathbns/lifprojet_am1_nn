# =============================================================================
# Fichier de constantes - Configuration du projet
# =============================================================================
import os

# Fonction pour obtenir le répertoire racine du projet (un niveau au-dessus de src/)
def get_project_root():
    """Retourne le chemin absolu vers la racine du projet."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = get_project_root()
SRC_ROOT = os.path.dirname(os.path.abspath(__file__))

# Chemins vers les données brutes (chemins absolus basés sur la racine du projet)
PGN_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'raw', 'pgns') + os.sep
RAW_DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'classic', 'games') + os.sep

# Chemins vers les données traitées
LABELED_DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'labeled') + os.sep
PREPROCESSED_DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'preprocessed', 'games') + os.sep
CNN_DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'CNN') + os.sep

# Chemins pour les résultats et fichiers temporaires
RESULTS_FOLDER = os.path.join(PROJECT_ROOT, 'results') + os.sep
TEMP_SVG_FOLDER = PROJECT_ROOT + os.sep
MODEL_WEIGHTS_FILE = os.path.join(SRC_ROOT, 'model_weights.weights.h5')
CLASS_INDICES_FILE = os.path.join(SRC_ROOT, 'class_indices.json')

# Compteur global pour le numérotage des images labellisées
image_counter = 0

# Dictionnaire de correspondance entre notation de pièces d'échecs et noms
# Utilisé pour convertir les pièces depuis la notation FEN vers les noms de classes
PIECE_LABELS = {
    'p': 'Pawn',
    'r': 'Rook',
    'n': 'Knight',
    'b': 'Bishop',
    'q': 'Queen',
    'k': 'King'
}

# Dictionnaire de correspondance entre noms de classes et notation FEN
# Utilisé pour convertir les prédictions du modèle en notation d'échecs standard
FEN_LABELS = {
    'Empty': '.',
    'Rook_White': 'R',
    'Rook_Black': 'r',
    'Knight_White': 'N',
    'Knight_Black': 'n',
    'Bishop_White': 'B',
    'Bishop_Black': 'b',
    'Queen_White': 'Q',
    'Queen_Black': 'q',
    'King_White': 'K',
    'King_Black': 'k',
    'Pawn_White': 'P',
    'Pawn_Black': 'p',
}

# Liste des classes de pièces dans l'ordre alphabétique
# Cet ordre doit correspondre à celui utilisé lors de l'entraînement du modèle
PIECE_CLASSES = [
    'Bishop_Black', 'Bishop_White', 'Empty', 'King_Black', 'King_White',
    'Knight_Black', 'Knight_White', 'Pawn_Black', 'Pawn_White',
    'Queen_Black', 'Queen_White', 'Rook_Black', 'Rook_White'
]

# Configuration pour l'entraînement du modèle
NUM_EPOCHS = 10
BATCH_SIZE = 16
IMAGE_SIZE = (300, 150)  # (height, width)

# Configuration pour le split des données
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

# Configuration pour le traitement d'images
SQUARE_LENGTH = 150  # Taille d'une case d'échecs en pixels
RESIZE_HEIGHT = 500  # Hauteur de référence pour le redimensionnement
