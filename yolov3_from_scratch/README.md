# YOLOv3 - You Only Look Once (Version 3)

Implémentation from scratch de l'architecture YOLOv3 avec Darknet-53 et Feature Pyramid Network (FPN).

![Architecture YOLOv3](architecture_yolov3.png)

## Description

YOLOv3 représente une évolution majeure de YOLO avec l'introduction de la détection multi-échelle. Cette architecture améliore considérablement la détection des petits objets grâce à son Feature Pyramid Network (FPN) qui prédit à trois échelles différentes.

**Innovations clés par rapport à YOLOv1:**
- **Détection multi-échelle:** 3 niveaux de prédiction (13×13, 26×26, 52×52)
- **Backbone amélioré:** Darknet-53 avec blocs résiduels
- **Meilleure précision:** Surtout pour les petits objets
- **Feature Pyramid Network:** Connexions skip pour combiner features de différentes résolutions
- **Anchors prédéfinis:** 9 anchors (3 par échelle)
- **Pas de fully connected:** Architecture entièrement convolutive

## Architecture

### 1. Backbone: Darknet-53

Le backbone Darknet-53 est composé de **53 couches convolutives** avec des blocs résiduels:

```
Input (416×416×3)
↓
Conv (3×3×32)
↓
Conv + Residual×1  (64 channels)   ──┐
↓                                    │
Conv + Residual×2  (128 channels)  ──┤
↓                                    │
Conv + Residual×8  (256 channels)  ──┤─→ Route 1 (vers Scale 3)
↓                                    │
Conv + Residual×8  (512 channels)  ──┤─→ Route 2 (vers Scale 2)
↓                                    │
Conv + Residual×4  (1024 channels) ──┘
↓
Neck (FPN)
```

**Caractéristiques:**
- **Residual Blocks:** Connexions résiduelles type ResNet pour stabilité
- **Batch Normalization:** Après chaque couche convolutive
- **LeakyReLU:** Activation (α=0.1)
- **Pas de pooling:** Stride=2 pour downsampling
- **Paramètres:** ~41 millions pour le backbone seul

### 2. Neck: Feature Pyramid Network (FPN)

Le FPN combine des features de différentes résolutions pour améliorer la détection:

```
Scale 1 (13×13) ────────────→ Prédiction 1 (Large objects)
     ↓ Upsample ×2
     + Concat Route 2
     ↓
Scale 2 (26×26) ────────────→ Prédiction 2 (Medium objects)
     ↓ Upsample ×2
     + Concat Route 1
     ↓
Scale 3 (52×52) ────────────→ Prédiction 3 (Small objects)
```

### 3. Head: Multi-Scale Detection

Trois têtes de détection indépendantes:

| Scale | Grid Size | Anchors | Output Shape | Détecte |
|-------|-----------|---------|--------------|---------|
| Scale 1 | 13×13 | (116,90), (156,198), (373,326) | 13×13×255 | Grands objets |
| Scale 2 | 26×26 | (30,61), (62,45), (59,119) | 26×26×255 | Objets moyens |
| Scale 3 | 52×52 | (10,13), (16,30), (33,23) | 52×52×255 | Petits objets |

**Output par échelle:** 3 anchors × (20 classes + 5 params) = 255 channels
- 5 params: (x, y, w, h, confidence)

**Paramètres totaux:** ~62 millions

## Dataset

### PASCAL VOC

Identique à YOLOv1, le modèle utilise PASCAL VOC avec **20 classes:**

```
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep,
sofa, train, tvmonitor
```

**Spécifications:**
- Images redimensionnées: 416×416 pixels (vs 448×448 pour YOLOv1)
- Format des labels: YOLO (x_center, y_center, width, height)
- Augmentation de données: rotation, flip, color jitter, etc.

### Téléchargement

```bash
# Via Kaggle API
kaggle datasets download -d aladdinpersson/pascal-voc-yolo

# Extraction dans le dossier data/
unzip pascal-voc-yolo.zip -d data/PASCAL_VOC/
```

**Structure attendue:**
```
data/PASCAL_VOC/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── train.csv
└── val.csv
```

## Fonction de perte

La loss YOLOv3 est calculée pour chaque échelle et combinée:

```python
Total Loss = Σ (Box Loss + Object Loss + No-Object Loss + Class Loss)
             scale ∈ {1,2,3}
```

### Composantes de la loss

1. **Box Loss (Localization):**
   - MSE sur les coordonnées (x, y)
   - MSE sur les dimensions (w, h)
   - Appliqué uniquement aux cellules contenant des objets

2. **Object Loss (Confidence):**
   - Binary Cross-Entropy sur le score de confiance
   - Pour les cellules contenant des objets

3. **No-Object Loss:**
   - Binary Cross-Entropy sur le score de confiance
   - Pour les cellules sans objet
   - Poids réduit (généralement 0.5)

4. **Class Loss:**
   - Binary Cross-Entropy sur les probabilités de classe
   - Multi-label (un objet peut appartenir à plusieurs classes)

### Caractéristiques

- **IoU-based assignment:** Chaque anchor est assigné à l'objet avec le meilleur IoU
- **Ignore threshold:** Anchors avec IoU > 0.5 sont ignorés (ni obj ni no-obj)
- **Multi-label classification:** Contrairement à YOLOv1 (single-label)

## Installation et utilisation

### Prérequis

```bash
pip install -r requirements.txt
```

**Dépendances principales:**
```
torch>=2.0.0
torchvision>=0.15.0
numpy
pandas
Pillow
tqdm
matplotlib
albumentations  # Pour l'augmentation de données
opencv-python
```

### Configuration

Le fichier `config.py` contient tous les hyperparamètres:

```python
# Dataset
DATASET = 'data/PASCAL_VOC'
IMAGE_SIZE = 416
NUM_CLASSES = 20

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100

# Detection
CONF_THRESHOLD = 0.05     # Seuil de confiance
NMS_IOU_THRESH = 0.45     # Seuil NMS
MAP_IOU_THRESH = 0.5      # Seuil mAP

# Anchors (width, height) normalisés
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],    # Scale 1
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],   # Scale 2
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],   # Scale 3
]
```

### Entraînement

```bash
# Entraînement depuis zéro
python train.py

# Reprendre depuis un checkpoint
python train.py --load_model --checkpoint checkpoints/checkpoint.pth.tar
```

### Structure des fichiers

```
yolov3_from_scratch/
├── model.py          # Architecture YOLOv3 (Darknet-53 + FPN)
├── loss.py           # Fonction de perte multi-échelle
├── dataset.py        # Chargement des données avec anchors
├── train.py          # Script d'entraînement
├── utils.py          # Utilitaires (IoU, NMS, mAP, visualisation)
├── config.py         # Configuration et hyperparamètres
├── requirements.txt  # Dépendances Python
└── README.md         # Cette documentation
```

## 📈 Entraînement et performance

### Spécifications d'entraînement

- **Durée:** ~8-10 heures sur GPU A100
- **Dataset:** PASCAL VOC 2007+2012
- **Batch size:** 32
- **Learning rate:** 1e-5 avec weight decay 1e-4
- **Optimizer:** Adam
- **Augmentation:** Rotation, flip, color jitter, affine transforms

### Métriques

- **mAP@0.5:** Mean Average Precision avec IoU threshold = 0.5
- **mAP@0.5:0.95:** mAP moyenné sur plusieurs seuils IoU
- **FPS:** Frames Per Second pour l'inférence

### Comparaison YOLOv1 vs YOLOv3

| Métrique | YOLOv1 | YOLOv3 | Amélioration |
|----------|--------|--------|--------------|
| mAP@0.5 | ~63% | ~74% | +11% |
| Petits objets | Faible | Bon | ⭐⭐⭐ |
| FPS (GPU) | ~45 | ~30 | -33% |
| Paramètres | 45M | 62M | +38% |
| Grid cells | 7×7 | 13×13 + 26×26 + 52×52 | Multi-échelle |

## 🔍 Fonctions utilitaires

### Intersection over Union (IoU)

```python
# IoU pour width/height (utilisé pour anchor matching)
iou_width_height(boxes1, boxes2)

# IoU complet avec coordonnées
intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint")
```

### Non-Maximum Suppression (NMS)

```python
non_max_suppression(
    predictions, 
    iou_threshold=0.45, 
    threshold=0.4, 
    box_format="midpoint"
)
```

### Mean Average Precision (mAP)

```python
mean_average_precision(
    pred_boxes, 
    true_boxes, 
    iou_threshold=0.5, 
    num_classes=20
)
```

### Conversion et visualisation

```python
# Convertir les prédictions de grille en bounding boxes
cells_to_bboxes(predictions, anchors, S, is_preds=True)

# Visualiser les résultats
plot_image(image, boxes)
plot_couple_examples(model, loader, threshold=0.6)

# Sauvegarder/charger le modèle
save_checkpoint(model, optimizer, filename="checkpoint.pth.tar")
load_checkpoint(checkpoint_file, model, optimizer, lr)
```

## 📚 Références académiques

**Article original YOLOv3:**
```
Redmon, J., & Farhadi, A. (2018).
"YOLOv3: An Incremental Improvement."
arXiv preprint arXiv:1804.02767.
```

**Lien:** [arXiv:1804.02767](https://arxiv.org/abs/1804.02767)

**Articles connexes:**
- YOLOv1: [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)
- YOLOv2/YOLO9000: [arXiv:1612.08242](https://arxiv.org/abs/1612.08242)
- Feature Pyramid Networks: [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)
- ResNet: [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

## Détails d'implémentation

### Residual Blocks

```python
class ResidualBlock(nn.Module):
    """Bloc résiduel avec connexion de saut"""
    def __init__(self, channels, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers.append(
                nn.Sequential(
                    CNN(channels, channels // 2, kernel_size=1),      # Réduction
                    CNN(channels // 2, channels, kernel_size=3, padding=1),  # Expansion
                )
            )
    
    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)  # Connexion résiduelle
        return x
```

### Scale Prediction

```python
class ScalePrediction(nn.Module):
    """Prédiction d'échelle pour la sortie YOLO"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNN(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNN(2 * in_channels, (num_classes + 5) * 3, kernel_size=1, bn_act=False),
        )
    
    def forward(self, x):
        # Reshape: [batch, 3, grid, grid, num_classes + 5]
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )
```

### Augmentation de données

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms = A.Compose([
    A.LongestMaxSize(max_size=int(IMAGE_SIZE * 1.1)),
    A.PadIfNeeded(min_height=int(IMAGE_SIZE * 1.1), 
                  min_width=int(IMAGE_SIZE * 1.1)),
    A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
    A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
    A.Affine(shear=15, rotate=20, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Blur(p=0.1),
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
    ToTensorV2(),
], bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]))
```

## Avantages et améliorations

### Avantages de YOLOv3

1. **Détection multi-échelle:**
   - Excellente performance sur petits, moyens et grands objets
   - Grilles de 52×52, 26×26 et 13×13 cellules

2. **Architecture résiduelle:**
   - Meilleure propagation du gradient
   - Entraînement plus stable et profond (53 couches)

3. **Feature Pyramid Network:**
   - Combine features de haute et basse résolution
   - Enrichit la représentation des features

4. **Classification multi-label:**
   - Un objet peut appartenir à plusieurs classes
   - Plus flexible que YOLOv1

5. **Anchors optimisés:**
   - 9 anchors adaptés aux différentes échelles
   - Meilleure couverture des ratios d'aspect

### Améliorations possibles

- **Architecture plus récente:** YOLOv4, YOLOv5, YOLOX
- **Attention mechanisms:** CBAM, SE-Net
- **Data augmentation avancée:** Mosaic, MixUp
- **Loss functions:** GIoU, DIoU, CIoU
- **Post-processing:** Soft-NMS, DIoU-NMS

## 📝Notes importantes

⚠️ **Checkpoints et modèles:**
- Les poids pré-entraînés ne sont pas inclus (fichiers volumineux)
- Le fichier `checkpoints/checkpoint.pth.tar` doit être téléchargé séparément
- Durée d'entraînement : ~8-10h sur GPU A100

⚠️ **Dataset:**
- Le dataset PASCAL VOC doit être téléchargé via Kaggle
- Environ 20GB d'espace disque nécessaire
- Structure de dossiers spécifique requise (voir section Dataset)

⚠️ **Ressources:**
- GPU recommandé (training impossible sur CPU)
- Minimum 16GB de RAM
- ~25GB d'espace disque total (dataset + checkpoints)
