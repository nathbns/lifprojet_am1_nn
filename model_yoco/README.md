# YOCO - You Only Chess Once

> Système complet de reconnaissance et d'analyse de positions d'échecs à partir d'une simple photo.

![Bannière YOCO](../public/banniere_yoco.jpg)

---

## Table des matières

1. [Objectifs](#objectifs)
2. [Installation](#installation)
3. [Utilisation](#utilisation)
4. [Organisation du code](#organisation-du-code)
5. [Pipeline technique](#pipeline-technique)
6. [Données et modèle](#données-et-modèle)
7. [Références](#références)

---

## Objectifs

YOCO est un pipeline end-to-end permettant de :

- **Détecter automatiquement** un échiquier dans une photo (angles, perspective, recadrage)
- **Reconnaître chaque case** du plateau (12 types de pièces + case vide = 13 classes)
- **Reconstruire la position** en notation FEN (Forsyth-Edwards Notation)
- **Générer une visualisation** de la position détectée (export PNG)

### Caractéristiques techniques

| Composant | Description |
|-----------|-------------|
| **Prétraitement** | Algorithmes SLID (détection de lignes) + LAPS (validation d'intersections) |
| **Architecture CNN** | 5 couches convolutives (16, 32, 64, 64, 64 filtres) + Dense 128 |
| **Classification** | 13 classes (6 pièces x 2 couleurs + case vide) |
| **Entrée modèle** | Images 300x150 pixels |
| **Sortie** | Notation FEN + rendu visuel PNG |

---

## Installation

### Prérequis

- Python 3.8+
- pip (gestionnaire de paquets Python)

### Étapes d'installation

```bash
# 1. Cloner le dépôt (si ce n'est pas déjà fait)
git clone https://github.com/nathbns/lifprojet_am1_nn.git
cd lifprojet_am1_nn/model_yoco

# 2. Créer un environnement virtuel (recommandé)
python -m venv .venv

# 3. Activer l'environnement virtuel
# Sur macOS/Linux :
source .venv/bin/activate
# Sur Windows :
.venv\Scripts\activate

# 4. Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales

| Package | Usage |
|---------|-------|
| `tensorflow` / `keras` | Entraînement et inférence du CNN |
| `opencv-contrib-python` | Traitement d'images, détection de lignes |
| `numpy` | Calculs matriciels |
| `scipy` | Clustering, transformations géométriques |
| `scikit-learn` | DBSCAN, métriques d'évaluation |
| `chess` | Manipulation de positions FEN et PGN |
| `wand` | Conversion SVG vers PNG pour la visualisation |
| `pyclipper` | Opérations géométriques sur polygones |

**Note :** Pour upload les poids de notre modeles https://huggingface.co/nathbns/yoco_tf

---

## Utilisation

### Inférence rapide (reconnaissance d'une position)

```bash
# Depuis le dossier model_yoco/
python src/main.py chemin/vers/votre_image.jpg
```

**Exemple :**
```bash
python src/main.py example1.jpg
```

**Sortie :**
- Position FEN affichée dans le terminal
- Image annotée sauvegardée dans `results/`

### Réentraîner le modèle

Si vous souhaitez réentraîner le modèle avec vos propres données :

#### 1. Préparer les images brutes
Placez vos photos dans `data/raw/<nom_partie>/<white|black>/`
- `white` : photos depuis le côté blanc
- `black` : photos depuis le côté noir

#### 2. Prétraiter les images
```bash
python src/preprocessing/image_processing.py
```
Génère les images recadrées dans `data/preprocessed/games/`

#### 3. Générer les labels automatiquement
Ajoutez les fichiers PGN correspondants dans `data/raw/pgns/`, puis :
```bash
python src/utils/labeling.py
```
Découpe chaque case et classifie dans `data/labeled/<classe>/`

#### 4. Créer le split train/val/test
```bash
python src/utils/data_splitting.py
```
Répartit les données dans `data/CNN/train|validation|test/<classe>/`

#### 5. Entraîner le modèle
```bash
python src/training/model_training.py
```
Sauvegarde les poids dans `src/model_weights.weights.h5`

---

## Organisation du code

```
model_yoco/
│
├── src/                          # Code source principal
│   ├── main.py                   # Point d'entrée - Pipeline d'inférence complet
│   ├── constant.py               # Configuration (chemins, tailles, classes)
│   ├── class_indices.json        # Mapping des indices de classes
│   ├── model_weights.weights.h5  # Poids du modèle entraîné
│   │
│   ├── detection/                # Module de détection du plateau
│   │   ├── line_detection.py     # SLID : détection de lignes (CLAHE + Canny + Hough)
│   │   ├── lattice_detection.py  # LAPS : validation des intersections
│   │   └── board_corners.py      # Extraction des 4 coins du plateau
│   │
│   ├── preprocessing/            # Module de prétraitement
│   │   └── image_processing.py   # Recadrage + correction de perspective
│   │
│   ├── training/                 # Module d'entraînement
│   │   └── model_training.py     # Création et entraînement du CNN
│   │
│   └── utils/                    # Utilitaires
│       ├── data_splitting.py     # Split train/val/test (80/10/10)
│       ├── image_transforms.py   # Transformations géométriques, homographie
│       └── labeling.py           # Labellisation automatique via PGN
│
├── deps/                         # Dépendances externes (adaptées)
│   ├── laps.py                   # Modèle LAPS (détection d'intersections)
│   └── geometry.py               # Fonctions géométriques auxiliaires
│
├── data/                         # Données (non versionnées, à créer)
│   ├── raw/                      # Images brutes + fichiers PGN
│   ├── preprocessed/             # Images après recadrage
│   ├── labeled/                  # Cases labellisées par classe
│   ├── CNN/                      # Données prêtes pour l'entraînement
│   └── laps_models/              # Poids du modèle LAPS
│
├── results/                      # Sorties d'inférence (FEN + images)
├── example1.jpg, example2.jpg    # Images d'exemple pour test
└── requirements.txt              # Dépendances Python
```

### Description des modules

| Module | Fichier | Fonction |
|--------|---------|----------|
| **Détection** | `line_detection.py` | Applique CLAHE, Canny, Hough pour extraire les lignes de l'échiquier |
| **Détection** | `lattice_detection.py` | Calcule les intersections et les valide via le réseau LAPS |
| **Détection** | `board_corners.py` | Identifie les 4 coins du plateau par scoring géométrique |
| **Prétraitement** | `image_processing.py` | Orchestre le pipeline complet de prétraitement |
| **Entraînement** | `model_training.py` | Définit l'architecture CNN et gère l'entraînement |
| **Utilitaires** | `labeling.py` | Aligne les cases avec les positions PGN pour créer les labels |
| **Utilitaires** | `data_splitting.py` | Répartit les données en train/val/test |
| **Utilitaires** | `image_transforms.py` | Applique l'homographie pour rectifier la perspective |

---

## Pipeline technique

Le pipeline transforme une photo brute en une position FEN en plusieurs étapes :

### 1. Détection de lignes (SLID)
- **CLAHE** : améliore le contraste local pour rendre les lignes visibles
- **Canny** : détecte les contours (transitions d'intensité)
- **Hough** : convertit les points de contour en équations de droites
- **Union-Find** : fusionne les segments colinéaires

### 2. Validation des intersections (LAPS)
- Calcule toutes les intersections entre lignes détectées
- Extrait un patch 21x21 autour de chaque candidat
- Le réseau LAPS filtre les vraies intersections
- Clustering hiérarchique pour éliminer les doublons

### 3. Extraction des coins
- **DBSCAN** : identifie le cluster principal de points
- Génération de quadrilatères candidats
- **Score géométrique** : aire, convexité, densité de points LAPS
- Sélection du meilleur quadrilatère

### 4. Rectification et découpage
- **Homographie** : transforme le quadrilatère en carré 1200x1200 pixels
- Découpage en 64 cases de 150x150 pixels

### 5. Classification CNN
- Chaque case est redimensionnée en 300x150 pixels
- Passage dans le CNN pour obtenir une des 13 classes
- Reconstruction de la notation FEN complète

### A propos du prétraitement

Le pipeline de prétraitement (SLID, LAPS, extraction de coins) est **inspiré de l'approche** du projet [laps](https://github.com/maciejczyzewski/neural-chessboard/tree/draft/deps)

Seuls les **poids pré-entraînés du modèle LAPS** (réseau de validation des intersections) proviennent du projet original. Tout le reste du code de prétraitement a été développé par nos soins pour fonctionner avec notre modèle CNN. Nous aurions pu ré-entrainer ce modele, mais étant donnée que c'étais un CNN léger nous aurions eu les memes resultat sans réelle apprentissage ou aboutissant supplémentaire pour notre projet.

---

## Données et modèle

### Dataset YOCO (Faite par nous memes)

- **670 photos** d'un échiquier réel prises sous différents angles
- **Labellisation automatique** via alignement avec les fichiers PGN
- **13 classes** : `Empty`, `Pawn_White`, `Pawn_Black`, `Rook_White`, `Rook_Black`, `Knight_White`, `Knight_Black`, `Bishop_White`, `Bishop_Black`, `Queen_White`, `Queen_Black`, `King_White`, `King_Black`

### Liens utiles

| Ressource | URL |
|-----------|-----|
| **Notre Dataset sur HuggingFace** | https://huggingface.co/datasets/nathbns/chess-yoco |
| **Notre Dataset sur Kaggle** | https://www.kaggle.com/datasets/nathanbensoussan/yoco-dataset-for-chess-pieces-classification |
| **Espace d'inférence** | https://huggingface.co/spaces/nathbns/yoco_first_version |
| **Visualisation prétraitement** | https://huggingface.co/spaces/nathbns/preprocess_yoco |

**Note :** Les modèles HuggingFace gratuits se mettent en veille après 48h d'inactivité. Un premier appel peut prendre plus de temps pour "réveiller" le modèle.

### Architecture du CNN

```
Input: (300, 150, 3)
    |
Conv2D(16, 3x3, ReLU) -> MaxPool(2x2)
    |
Conv2D(32, 3x3, ReLU) -> MaxPool(2x2)
    |
Conv2D(64, 3x3, ReLU) -> MaxPool(2x2)
    |
Conv2D(64, 3x3, ReLU) -> MaxPool(2x2)
    |
Conv2D(64, 3x3, ReLU) -> MaxPool(2x2)
    |
Flatten -> Dense(128, ReLU) -> Dense(13, Softmax)
    |
Output: classe parmi 13
```

**Hyperparamètres :**
- Optimiseur : RMSprop (lr=0.001)
- Loss : Categorical CrossEntropy
- Epoques : 10
- Batch size : 16
- Split : 80% train / 10% val / 10% test

---

## Auteurs

- **Nathan BEN SOUSSAN**
- **Mehdi MOUJIB**

---
