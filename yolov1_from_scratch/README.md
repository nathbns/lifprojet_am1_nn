# YOLOv1 - You Only Look Once (Version 1)

Implémentation from scratch de l'architecture YOLOv1 basée sur l'article original de Redmon et al. (2016).

![Architecture YOLOv1](architecture_yolov1.png)

## Description

Nous avons suivis la methode et démarche décrite de la version papier sortie en 2016, de la manière la plus proche possible.

**Principe de fonctionnement:**
1. L'image est divisée en une grille S×S (7×7 dans notre cas)
2. Chaque cellule prédit B boîtes englobantes (2 dans notre cas) et leurs scores de confiance
3. Chaque cellule prédit également les probabilités de classe
4. Les prédictions finales sont obtenues après Non-Maximum Suppression (NMS)

## Architecture

L'architecture de notre YOLOv1 "from scratch" se compose de:

### Backbone (Darknet-24)
- **24 couches convolutives** pour l'extraction de features
- **4 couches de max-pooling** pour la réduction dimensionnelle
- **Activation:** LeakyReLU (α=0.1) pour toutes les couches
- **Normalisation:** Batch Normalization après chaque convolution (Non présente dans la version de 2016, mais nous avons souhaitez l'ajouter afin d'avoir une amélioration)

### Detection Head
- **2 couches fully connected** (4096→496→1470)
- **Dropout** (taux: 0.0 dans cette implémentation)
- **Output:** Tensor 7×7×30
  - 20 probabilités de classe (PASCAL VOC)
  - 2 boîtes × (4 coordonnées + 1 confiance) = 10 valeurs

### Détails techniques
```
Input: 448×448×3
↓
24 Convolutional Layers
↓
Flatten: 7×7×1024 → 50176
↓
FC1: 50176 → 4096
↓
FC2: 4096 → 1470
↓
Reshape: 7×7×30
```

**Paramètres totaux:** ~45 millions

## 📊 Dataset

### PASCAL VOC
Le modèle est entraîné sur le dataset PASCAL VOC qui contient **20 classes d'objets:**

```
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep,
sofa, train, tvmonitor
```

**Spécifications:**
- Images redimensionnées: 448×448 pixels
- Format des labels: YOLO (x_center, y_center, width, height)
- Split: 80% train / 20% validation

### Téléchargement du dataset

Le dataset PASCAL VOC peut être téléchargé via Kaggle et c'est ce que nous avons fait pour notre entrainement:

```bash
pip install kaggle

# telecharger le dataset
kaggle datasets download -d aladdinpersson/pascal-voc-yolo

unzip pascal-voc-yolo.zip -d data/
```

## Fonction de perte (loss function)

La loss YOLOv1 est une combinaison pondérée de plusieurs composantes:

```
Total Loss = λcoord × Localization Loss 
           + Confidence Loss (obj) 
           + λnoobj × Confidence Loss (no obj)
           + Classification Loss
```

Où:
- **λcoord = 5:** Poids pour les erreurs de localisation
- **λnoobj = 0.5:** Poids pour les cellules sans objet
- **Localization Loss:** MSE sur (x, y, √w, √h)
- **Confidence Loss:** MSE sur le score de confiance
- **Classification Loss:** MSE sur les probabilités de classe

## Installation et utilisation

### Prérequis

```bash
pip install -r requirements.txt
```

**Dépendances principales:**
- PyTorch >= 2.0.0
- torchvision
- numpy
- pandas
- Pillow
- tqdm

### Entraînement

```bash
python3 train.py
```

**Hyperparamètres par défaut:**
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 100
- Weight decay: 0
- Optimizer: Adam

### Structure des fichiers

```
yolov1_from_scratch/
├── model.py          # Archi du modèle YOLOv1
├── loss.py           # Fonction de perte YOLOv1
├── dataset.py        # Chargement des données PASCAL VOC
├── train.py          # Script d'entraînement
├── utils.py          # Fonctions utilitaires (IoU, NMS, mAP, etc.)
├── requirements.txt  # Dépendances
└── README.md         # doc
```

## Résultats

### Entraînement
- **Durée:** ~3 heures sur GPU A100 (Google Colab)
- **Dataset:** PASCAL VOC 2007+2012
- **Métrique:** Mean Average Precision (mAP) @ IoU 0.5

## Pour essayer notre modèle

### Hugging Face Space
Le modèle entraîné est disponible en ligne:
- **Demo interactive:** https://huggingface.co/spaces/nathbns/yolo1_from_scratch

### ou sur notre App web
Une interface web complète est disponible dans le dossier `webapp/`:
- **site web:** https://yoco-ochre.vercel.app (onglet yolo)

## Fonctions utilitaires

Le fichier `utils.py` contient de nombreuses fonctions essentielles:

### Intersection over Union (IoU)
```python
iou(boxes_preds, boxes_labels, box_format="midpoint")
```

### Non-Maximum Suppression (NMS)
```python
non_max_suppression(predictions, iou_threshold=0.5, threshold=0.4)
```

### Mean Average Precision (mAP)
```python
mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5)
```

### Conversion de coordonnées
```python
cellboxes_to_boxes(predictions, S=7)  # Grille → Image
```

### Visualisation
```python
plot_image(image, boxes)  # Affiche l'image avec les bounding boxes
save_checkpoint(state, filename="checkpoint.pth.tar")  # Sauvegarde
```

## Références académiques

**Article original:**
```
Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016).
"You only look once: Unified, real-time object detection."
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
```

**Lien:** [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)


## Architecture détaillée

### Couches convolutives

| Layer | Filters | Size | Stride | Output |
|-------|---------|------|--------|--------|
| Conv1 | 64 | 7×7 | 2 | 224×224×64 |
| MaxPool1 | - | 2×2 | 2 | 112×112×64 |
| Conv2 | 192 | 3×3 | 1 | 112×112×192 |
| MaxPool2 | - | 2×2 | 2 | 56×56×192 |
| Conv3-6 | 128→512 | mix | 1 | 56×56×512 |
| MaxPool3 | - | 2×2 | 2 | 28×28×512 |
| Conv7-14 | 256↔512 | mix | 1 | 28×28×512 |
| Conv15-16 | 512→1024 | mix | 1 | 28×28×1024 |
| MaxPool4 | - | 2×2 | 2 | 14×14×1024 |
| Conv17-20 | 512↔1024 | mix | 1 | 14×14×1024 |
| Conv21-24 | 1024 | 3×3 | 1-2 | 7×7×1024 |

### Les params d'entraînement que nous avons utilisé

```python
SPLIT_SIZE = 7        # Grille 7×7
NUM_BOXES = 2         # 2 boîtes par cellule
NUM_CLASSES = 20      # Classes PASCAL VOC

# Loss
LAMBDA_COORD = 5      # Poids localisation
LAMBDA_NOOBJ = 0.5    # Poids background

# Training
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 100
```

## Évolutions

Cette implémentation YOLOv1 a servi de base pour notre version de YOLOv3, voir `../yolov3_from_scratch/`.

## **Notes**

- Les poids du modèle entraîné ne sont pas inclus dans ce repository (taille importante)
- Téléchargez-les depuis notre [Hugging Face Space](https://huggingface.co/nathbns/yolov1_from_scratch)
- Le dataset PASCAL VOC doit être téléchargé séparément (voir section Dataset)

