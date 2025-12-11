"""
Module de détection des points du réseau (lattice) de l'échiquier.

Ce module implémente l'algorithme LAPS (Lattice Points) adapté
pour détecter les intersections valides des lignes de l'échiquier.

L'approche utilise :
1. Calcul des intersections entre toutes les lignes détectées
2. Validation de chaque intersection avec un classificateur
3. Clustering des points proches pour éliminer les doublons
"""

import sys
import os
import numpy as np
import cv2
import collections
from typing import List, Tuple, Optional
import scipy.spatial
import scipy.cluster.hierarchy

# Configuration des imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)

import deps.geometry as geometry

Point = Tuple[float, float]
LineSegment = List[List[int]]

# Chargement du modèle de classification (optionnel)
_MODEL_LOADED = False
_NEURAL_MODEL = None

def _load_classification_model():
    """Charge le modèle de classification des intersections."""
    global _MODEL_LOADED, _NEURAL_MODEL
    
    if _MODEL_LOADED:
        return _NEURAL_MODEL
    
    model_path = os.path.join(project_root, 'data', 'laps_models', 'laps.h5')
    
    try:
        import tensorflow as tf
        if os.path.exists(model_path):
            _NEURAL_MODEL = tf.keras.models.load_model(model_path, compile=False)
            from tensorflow.keras.optimizers import RMSprop
            _NEURAL_MODEL.compile(
                RMSprop(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['categorical_accuracy']
            )
        else:
            # Essaie de charger depuis deps
            from deps.laps import model as _NEURAL_MODEL
    except Exception as e:
        print(f"Avertissement: Impossible de charger le modèle LAPS: {e}")
        _NEURAL_MODEL = None
    
    _MODEL_LOADED = True
    return _NEURAL_MODEL


def yoco_find_line_intersections(lines: List[LineSegment]) -> List[Point]:
    """
    Trouve toutes les intersections entre les lignes.
    
    Utilise l'algorithme de géométrie pour calculer efficacement
    toutes les intersections entre segments.
    
    Args:
        lines: Liste des segments de lignes
        
    Returns:
        Liste des points d'intersection
    """
    # Convertit au format attendu par le module geometry
    segments = [((a[0], a[1]), (b[0], b[1])) for a, b in lines]
    
    # Calcule les intersections
    intersections = geometry.isect_segments(segments)
    
    return intersections


def yoco_cluster_nearby_points(
    points: List[Point],
    max_distance: float = 10.0
) -> List[Point]:
    """
    Regroupe les points proches en un seul point (leur moyenne).
    
    Utilise le clustering hiérarchique pour fusionner les points
    qui sont trop proches les uns des autres.
    
    Args:
        points: Liste des points à regrouper
        max_distance: Distance maximale pour fusionner deux points
        
    Returns:
        Liste des points après clustering
    """
    if len(points) < 2:
        return list(points)
    
    points_array = np.array(points)
    
    # Calcule la matrice de distances
    distances = scipy.spatial.distance.pdist(points_array)
    
    # Clustering hiérarchique
    linkage = scipy.cluster.hierarchy.single(distances)
    clusters = scipy.cluster.hierarchy.fcluster(linkage, max_distance, 'distance')
    
    # Regroupe les points par cluster
    cluster_points = collections.defaultdict(list)
    for i, cluster_id in enumerate(clusters):
        cluster_points[cluster_id].append(points[i])
    
    # Calcule le centre de chaque cluster
    centroids = []
    for points_in_cluster in cluster_points.values():
        arr = np.array(points_in_cluster)
        centroid = (np.mean(arr[:, 0]), np.mean(arr[:, 1]))
        centroids.append(centroid)
    
    return centroids


def yoco_extract_intersection_patch(
    image: np.ndarray,
    center: Point,
    patch_size: int = 10
) -> Optional[np.ndarray]:
    """
    Extrait une petite région autour d'un point d'intersection.
    
    Cette région sera analysée pour déterminer si c'est une
    intersection valide de l'échiquier.
    
    Args:
        image: Image source
        center: Centre de la région à extraire
        patch_size: Demi-taille de la région
        
    Returns:
        Région extraite ou None si invalide
    """
    x, y = int(center[0]), int(center[1])
    height, width = image.shape[:2]
    
    # Calcule les bornes
    x1 = max(0, x - patch_size - 1)
    x2 = max(0, x + patch_size)
    y1 = max(0, y - patch_size)
    y2 = max(0, y + patch_size + 1)
    
    # Vérifie que la région est valide
    if x2 <= x1 or y2 <= y1:
        return None
    
    patch = image[y1:y2, x1:x2]
    
    if patch.shape[0] <= 0 or patch.shape[1] <= 0:
        return None
    
    return patch


def yoco_validate_intersection_geometric(patch: np.ndarray) -> Tuple[bool, float]:
    """
    Valide une intersection avec une méthode géométrique.
    
    Recherche 4 contours quadrilatéraux dans la région,
    ce qui indique une intersection de lignes d'échiquier.
    
    Args:
        patch: Région autour de l'intersection
        
    Returns:
        Tuple (est_valide, score_de_confiance)
    """
    # Prétraitement
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    edges = cv2.Canny(binary, 0, 255)
    
    # Dilate pour connecter les contours
    dilated = cv2.dilate(edges, None)
    
    # Ajoute une bordure blanche
    bordered = cv2.copyMakeBorder(
        dilated, 1, 1, 1, 1,
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    bordered = cv2.bitwise_not(bordered)
    
    # Trouve les contours
    contours, _ = cv2.findContours(bordered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Compte les quadrilatères
    quad_count = 0
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        approx = cv2.approxPolyDP(contour, 0.1 * cv2.arcLength(contour, True), True)
        
        if len(approx) == 4 and radius < 14:
            quad_count += 1
    
    # 4 quadrilatères = intersection valide
    return quad_count == 4, float(quad_count) / 4.0


def yoco_validate_intersection_neural(patch: np.ndarray) -> Tuple[bool, float]:
    """
    Valide une intersection avec le réseau de neurones.
    
    Le modèle a été entraîné pour classifier les régions
    comme intersections valides ou non.
    
    Args:
        patch: Région autour de l'intersection
        
    Returns:
        Tuple (est_valide, score_de_confiance)
    """
    model = _load_classification_model()
    
    if model is None:
        # Fallback sur la méthode géométrique
        return yoco_validate_intersection_geometric(patch)
    
    # Prétraite l'image
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    edges = cv2.Canny(binary, 0, 255)
    resized = cv2.resize(edges, (21, 21), interpolation=cv2.INTER_CUBIC)
    
    # Prépare l'entrée
    input_data = np.where(resized > 127, 1, 0).reshape(-1, 21, 21, 1)
    
    # Prédit
    prediction = model.predict(input_data, verbose=0)
    prob_valid = prediction[0][0]
    prob_invalid = prediction[0][1]
    
    # Critères de validation
    is_valid = (prob_valid > prob_invalid and 
                prob_invalid < 0.03 and 
                prob_valid > 0.975)
    
    return is_valid, float(prob_valid)


def yoco_validate_intersection(
    image: np.ndarray,
    point: Point,
    patch_size: int = 10
) -> Tuple[bool, float]:
    """
    Valide si un point est une intersection valide de l'échiquier.
    
    Combine la validation géométrique et neuronale pour
    une meilleure précision.
    
    Args:
        image: Image de l'échiquier
        point: Point à valider
        patch_size: Taille de la région à analyser
        
    Returns:
        Tuple (est_valide, score_de_confiance)
    """
    # Extrait la région
    patch = yoco_extract_intersection_patch(image, point, patch_size)
    
    if patch is None:
        return False, 0.0
    
    # Essaie d'abord la validation géométrique
    is_valid_geo, score_geo = yoco_validate_intersection_geometric(patch)
    
    if is_valid_geo:
        return True, score_geo
    
    # Sinon, utilise le réseau de neurones
    return yoco_validate_intersection_neural(patch)


def yoco_detect_lattice_points(
    image: np.ndarray,
    lines: List[LineSegment],
    patch_size: int = 10
) -> List[Point]:
    """
    Détecte les points du réseau de l'échiquier.
    
    Pipeline complet :
    1. Trouve toutes les intersections des lignes
    2. Valide chaque intersection
    3. Regroupe les points proches
    
    Args:
        image: Image BGR de l'échiquier
        lines: Lignes détectées de l'échiquier
        patch_size: Taille des régions pour la validation
        
    Returns:
        Liste des points du réseau validés
        
    Example:
        >>> image = cv2.imread("chessboard.jpg")
        >>> lines = yoco_detect_chessboard_lines(image)
        >>> points = yoco_detect_lattice_points(image, lines)
        >>> print(f"Détecté {len(points)} points")
    """
    # Étape 1: Trouve toutes les intersections
    all_intersections = yoco_find_line_intersections(lines)
    
    # Étape 2: Valide chaque intersection
    valid_points = []
    
    for point in all_intersections:
        # Convertit en entiers
        int_point = (int(point[0]), int(point[1]))
        
        # Ignore les points hors de l'image
        if int_point[0] < 0 or int_point[1] < 0:
            continue
        
        height, width = image.shape[:2]
        if int_point[0] >= width or int_point[1] >= height:
            continue
        
        # Valide l'intersection
        is_valid, _ = yoco_validate_intersection(image, int_point, patch_size)
        
        if is_valid:
            valid_points.append(int_point)
    
    # Étape 3: Regroupe les points proches
    clustered_points = yoco_cluster_nearby_points(valid_points)
    
    return clustered_points