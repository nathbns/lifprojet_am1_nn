# YOCO - You Only Chess Once

Système complet de reconnaissance et d'analyse de positions d'échecs à partir d'une simple photo. Pipeline end-to-end : prétraitement du plateau, génération de labels depuis des parties PGN, entraînement d'un CNN et export de la position en notation FEN + rendu visuel.

![yoco](../public/banniere_yoco.jpg)

## Objectif
- Détecter automatiquement un échiquier dans une photo (angles, perspective, recadrage).
- Reconnaître chaque case (12 pièces + case vide = 13 classes).
- Reconstituer la position en notation FEN et produire une image annotée de sortie.

## Pipeline technique
1) **Prétraitement (vision)**  
   - Détection de lignes via SLID (`detection/line_detection.py`).  
   - Points d’intersection (LAPS) puis coins internes du plateau (`detection/lattice_detection.py`, `detection/board_corners.py`).  
   - Padding + transformation de perspective pour obtenir un échiquier carré (`utils/image_transforms.py`).  
2) **Labellisation automatique**  
   - Extraction des cases depuis les images prétraitées et alignement avec les coups PGN (`utils/labeling.py`).  
   - Génération des dossiers par classe dans `data/labeled/`.  
3) **Split & Dataset CNN**  
   - Répartition train/val/test dans `data/CNN/` (`utils/data_splitting.py`).  
4) **Entraînement**  
   - CNN 5×Conv (16,32,64,64,64 filtres) + dense 128, input `IMAGE_SIZE=(300,150)` (`training/model_training.py`).  
   - Sauvegarde des poids dans `src/model_weights.weights.h5` et des indices de classes `src/class_indices.json`.  
5) **Inférence & Export**  
   - Prétraitement → classification case par case → reconstruction FEN (`src/main.py`).  
   - Conversion FEN → SVG → PNG pour visualisation finale.

## Données & modèles
- **Dataset YOCO (670 photos)** : https://huggingface.co/datasets/nathbns/chess-yoco  
- **Espace inference** : https://huggingface.co/spaces/nathbns/yoco_first_version  
- 13 classes : `6 pièces × 2 couleurs + Empty`.
- Images de référence prises sur un échiquier réel (perspectives multiples).

## Détails algorithmiques (prétraitement)
- **SLID (Segment Let-It-Draw)** : pipeline CLAHE + Canny + Hough probabiliste pour extraire un grand nombre de segments, fusion/filtrage pour conserver les lignes structurantes du plateau.  
- **LAPS (Line-Augmented Patch Scoring)** : détecte et valide les points d’intersection des lignes. Nous réutilisons un modèle LAPS pré-entraîné existant (chargé depuis `data/laps_models/laps.h5` ou `deps/laps`) — ce n’est pas un modèle entraîné par nous.  
- **Coins internes & polygone convexe** : score géométrique (aire, densité de points, convexité) pour sélectionner le quadrilatère le plus plausible comme plateau (`detection/board_corners.py`).  
- **Padding & perspective** : élargit légèrement le polygone puis applique une homographie pour obtenir un échiquier carré 8×8 à échelle fixe (`utils/image_transforms.py`).  

## Installation rapide
```bash
cd model_yoco
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Inférence locale
```bash
# Depuis la racine du projet model_yoco
python src/main.py path/vers/une_image.jpg
# Résultats sauvegardés dans results/ avec rendu PNG + FEN affiché en console
```
Prérequis : le fichier de poids `src/model_weights.weights.h5` est déjà présent. Sinon, réentraîner (voir section suivante).

## Réentraîner le modèle
1) **Prétraiter les photos**  
   - Placez vos images brutes dans `data/raw/<game>/<orig|rev>/`.  
   - Lancez `python src/preprocessing/image_processing.py` (adapter le `__main__` ou appeler `yoco_preprocess_chess_games_list`).  
   - Les recadrages sont écrits dans `data/preprocessed/games/`.
2) **Générer les labels**  
   - Ajoutez les PGN dans `data/raw/pgns/`.  
   - Exécutez `python src/utils/labeling.py` pour découper chaque case et classer dans `data/labeled/`.
3) **Splitter train/val/test**  
   - `python src/utils/data_splitting.py` → remplit `data/CNN/train|validation|test/<classe>/`.
4) **Entraîner**  
   - `python src/training/model_training.py` (modèle Sequential Keras).  
   - Poids exportés dans `src/model_weights.weights.h5`, indices de classes dans `src/class_indices.json`.
5) **Tester / Inférer**  
   - Reprenez la commande d’inférence ci-dessus sur vos nouvelles images.

## Arborescence (résumé)
```
model_yoco/
├── src/
│   ├── main.py                   # Pipeline d'inférence FEN + PNG
│   ├── constant.py               # Config chemins, tailles, classes
│   ├── detection/                # SLID, LAPS, coins du plateau
│   ├── preprocessing/            # Recadrage + perspective
│   ├── training/                 # Génération des data loaders + CNN
│   └── utils/                    # Split, transforms, labeling
├── data/                         # raw/ preprocessed/ labeled/ CNN/
├── results/                      # Sorties inférence
├── example1.jpg / example2.jpg   # Exemples visuels
└── requirements.txt
```

## Notes & conseils
- Le modèle HuggingFace peut se mettre en veille (gratuit) : faire un appel pour le “réchauffer” avant une démo.
- Garder `IMAGE_SIZE`, `PIECE_CLASSES` et l’ordre des dossiers de classes synchronisés entre entraînement et inférence.
- Si la détection de coins échoue, vérifier la luminosité/contraste ou réessayer avec une photo plus centrée.

## Références
- Algorithmes SLID/LAPS pour la détection de grilles (implémentations dans `detection/`).
- Utilisation du modele laps pour la detection des grilles: https://github.com/maciejczyzewski/neural-chessboard

## Auteurs
- Nathan BEN SOUSSAN  
- Mehdi MOUJIB