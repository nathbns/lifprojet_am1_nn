# YOCO - You Only Chess Once

Application Next.js pour la détection et l'analyse de positions d'échecs à partir d'images.

## Fonctionnalités

- **Détection YOLO** : Détection d'objets avec les modèles YOLOv1 et YOLOv3
- **Reconnaissance d'échiquier** : Détection de positions d'échecs à partir d'images
- **Correction interactive** : Interface permettant de corriger manuellement les positions détectées
- **Calcul de précision** : Calcul automatique du pourcentage de réussite

## Technologies

- **Next.js 15** avec App Router
- **React 19**
- **TypeScript**
- **Tailwind CSS 4**
- **Chess.js** pour la logique d'échecs
- **React Chessboard** pour l'affichage des échiquiers
- **Gradio Client** pour l'intégration avec les modèles ML

## Démarrage

```bash
# Installation des dépendances
bun install

# Développement
bun dev

# Build de production
bun build

# Démarrage en production
bun start
```

L'application sera accessible sur [http://localhost:3000](http://localhost:3000)

## Ajout de composants Shadcn

```bash
bunx shadcn@latest add [component-name]
```

## Structure du projet

```
webapp/
├── app/              # Pages et routes Next.js
│   ├── chess/       # Page de détection d'échiquier
│   ├── yolo/        # Page de détection YOLO
│   └── docs/        # Page de documentation
├── components/       # Composants React réutilisables
│   └── ui/          # Composants UI Shadcn
├── lib/             # Utilitaires et clients API
└── public/          # Assets statiques
```