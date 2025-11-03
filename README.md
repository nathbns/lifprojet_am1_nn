# LifProjet - AM1 - YOCO

## Yolo Version 1 
- On suit l'article: https://arxiv.org/pdf/1506.02640 
- Recoder de zero, entrainer sur la dataset PASCAL VOC (telecharge via kaggle). 
- Entrainer 3H sur colab avec le GPU A100.

### Pour l'essayer:
Aller sur mon compte HF: https://huggingface.co/spaces/nathbns/yolo1_from_scratch \
Ou bien directement aller sur le site qui est hebergé sur vercel : https://yoco-ochre.vercel.app

## Yolo Version 3
Première version entrainer, je n'ai pas encore publié le code (prochainement...)

## Yoco (Prochainement commit du code (model, train, preprocess de l'image, etc...))
### Dataset utilisé (aucune créer de toute pièce.)
Première entrainement sur 670 images (335 prise vue des blanc / noir) de mon echiquier pris en photo.
La **dataset** est publié sur huggingface (Ne pas hesiter a liker, déjà **18 téléchargement en - de 24h!! 🤗**): https://huggingface.co/datasets/nathbns/chess-yoco

### Où l'essayer ?
- Sur notre app web onglet Chess (icone de la pièce fou)
- Sur le space HuggingFace: https://huggingface.co/spaces/nathbns/yoco_first_version
(Par ailleurs j'ai aussi fais un space pour visualiser le preprocess de l'image avant la detection: https://huggingface.co/spaces/nathbns/preprocess_yoco)

# Pour lancer l'application web
prerequis: installer bun
```bash
cd webapp && bun i 
```
et ensuite
```bash 
bun run dev
```
