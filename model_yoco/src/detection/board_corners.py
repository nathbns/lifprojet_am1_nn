"""
Module de détection des coins de l'échiquier.

Ce module implémente l'algorithme de détection des quatre coins
de l'échiquier à partir des points du réseau et des lignes détectées.

L'approche utilise :
1. Clustering des points pour identifier le groupe principal
2. Sélection des lignes candidates pour les bords
3. Scoring des quadrilatères candidats
4. Sélection du meilleur quadrilatère
"""

import sys
import os
import numpy as np
import cv2
import math
import itertools
import collections
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Configuration des imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from detection.lattice_detection import yoco_find_line_intersections, yoco_cluster_nearby_points

# Import optionnel pour le padding
try:
    import pyclipper
    PYCLIPPER_AVAILABLE = True
except ImportError:
    PYCLIPPER_AVAILABLE = False
    print("Avertissement: pyclipper non disponible, padding désactivé")

try:
    import sklearn.cluster
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.path
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

import scipy.spatial

# Type aliases
Point = Tuple[float, float]
IntPoint = List[int]
LineSegment = List[IntPoint]
Polygon = List[IntPoint]


def yoco_normalize_points(points: List[Point]) -> List[IntPoint]:
    """Convertit les points en entiers."""
    return [[int(a), int(b)] for a, b in points]


def yoco_filter_valid_points(
    points: List[IntPoint],
    image_shape: Tuple[int, ...]
) -> List[IntPoint]:
    """Filtre les points qui sont dans les limites de l'image."""
    height, width = image_shape[:2]
    valid = []
    for pt in points:
        if 0 <= pt[0] <= width and 0 <= pt[1] <= height:
            valid.append(pt)
    return valid


def yoco_sort_points_clockwise(points: List[IntPoint]) -> List[IntPoint]:
    """
    Trie les points dans l'ordre horaire autour de leur centre.
    
    Args:
        points: Liste de points à trier
        
    Returns:
        Points triés dans l'ordre horaire
    """
    if len(points) < 3:
        return points
    
    # Calcule le centre
    center_x = sum(p[0] for p in points) / len(points)
    center_y = sum(p[1] for p in points) / len(points)
    
    # Fonction de tri par angle
    def angle_from_center(point):
        return (math.atan2(point[0] - center_x, point[1] - center_y) + 2 * math.pi) % (2 * math.pi)
    
    return sorted(points, key=angle_from_center)


def yoco_remove_duplicate_items(items: List) -> List:
    """Supprime les doublons d'une liste en préservant l'ordre."""
    seen = set()
    result = []
    for item in items:
        item_tuple = tuple(map(tuple, item)) if isinstance(item[0], list) else tuple(item)
        if item_tuple not in seen:
            seen.add(item_tuple)
            result.append(item)
    return result


def yoco_calculate_point_to_line_distance(
    line: LineSegment,
    point: IntPoint
) -> float:
    """Calcule la distance perpendiculaire d'un point à une ligne."""
    line_vec = np.array(line[1]) - np.array(line[0])
    point_vec = np.array(point) - np.array(line[0])
    
    line_length = np.linalg.norm(line_vec)
    if line_length < 1e-8:
        return np.linalg.norm(point_vec)
    
    return abs(np.cross(line_vec, point_vec)) / line_length


def yoco_calculate_distance(p1: IntPoint, p2: IntPoint) -> float:
    """Calcule la distance euclidienne entre deux points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def yoco_score_quadrilateral(
    corners: np.ndarray,
    lattice_points: List[IntPoint],
    centroid: Point,
    alpha: float = 5,
    beta: float = 2
) -> float:
    """
    Calcule un score pour un quadrilatère candidat.
    
    Le score évalue :
    - Nombre de points du réseau contenus
    - Surface du quadrilatère
    - Régularité de la forme
    - Position du centre
    
    Args:
        corners: 4 coins du quadrilatère (numpy array)
        lattice_points: Points du réseau détectés
        centroid: Centre du nuage de points
        alpha: Paramètre de tolérance pour la taille
        beta: Paramètre de tolérance pour les points
        
    Returns:
        Score du quadrilatère (plus élevé = meilleur)
    """
    # Vérifie la surface minimale
    area = cv2.contourArea(corners)
    min_area = (4 * alpha * alpha) * 5
    
    if area < min_area:
        return 0.0
    
    # Compte les points contenus
    gamma = alpha / 1.5
    
    if PYCLIPPER_AVAILABLE and MATPLOTLIB_AVAILABLE:
        try:
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(corners.tolist(), pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
            expanded = pco.Execute(gamma)
            
            if not expanded:
                return 0.0
            
            path = matplotlib.path.Path(expanded[0])
            contained = path.contains_points(lattice_points)
            points_inside = min(np.count_nonzero(contained), 49)
        except:
            # Méthode alternative sans pyclipper
            points_inside = 0
            for pt in lattice_points:
                if cv2.pointPolygonTest(corners, tuple(pt), False) >= 0:
                    points_inside += 1
            points_inside = min(points_inside, 49)
    else:
        # Méthode alternative
        points_inside = 0
        for pt in lattice_points:
            if cv2.pointPolygonTest(corners, tuple(pt), False) >= 0:
                points_inside += 1
        points_inside = min(points_inside, 49)
    
    # Vérifie le nombre minimum de points
    min_points = min(len(lattice_points), 49) - 2 * beta - 1
    if points_inside < min_points:
        return 0.0
    
    A = points_inside  # Nombre de points
    B = area           # Surface
    
    if A == 0 or B == 0:
        return 0.0
    
    # Calcule la distance au centroïde
    corners_center = np.mean(corners, axis=0)
    G = np.linalg.norm(np.array(centroid) - corners_center)
    
    # Calcule l'erreur de forme (distance des points aux bords)
    E = 0
    F = 0
    edges = [
        [corners[0], corners[1]],
        [corners[1], corners[2]],
        [corners[2], corners[3]],
        [corners[3], corners[0]]
    ]
    
    for edge in edges:
        edge_length = yoco_calculate_distance(edge[0], edge[1])
        for pt in lattice_points:
            if cv2.pointPolygonTest(corners, tuple(pt), False) >= 0:
                dist = yoco_calculate_point_to_line_distance(edge, pt)
                if dist < gamma:
                    E += dist
                    F += 1
    
    if F == 0:
        return 0.0
    
    E /= F
    
    # Score final
    C = 1 + (E / A) ** (1/3)  # Régularité
    D = 1 + (G / A) ** (1/5)  # Centroïde
    
    score = (A ** 4) / ((B ** 2) * C * D)
    
    return score


def yoco_cluster_lattice_points(
    points: List[IntPoint],
    alpha: float
) -> List[IntPoint]:
    """
    Filtre les points aberrants avec DBSCAN.
    
    Args:
        points: Points à filtrer
        alpha: Paramètre de distance
        
    Returns:
        Points du cluster principal
    """
    if not SKLEARN_AVAILABLE or len(points) < 3:
        return points
    
    points_array = np.array(points)
    
    # Clustering DBSCAN
    clustering = sklearn.cluster.DBSCAN(eps=alpha * 4).fit(points_array)
    
    # Trouve le plus grand cluster
    clusters = collections.defaultdict(list)
    for i, label in enumerate(clustering.labels_):
        if label != -1:  # Ignore le bruit
            clusters[label].append(points[i])
    
    if not clusters:
        return points
    
    # Retourne le plus grand cluster
    largest = max(clusters.values(), key=len)
    
    return largest if len(largest) > len(points) / 2 else points


def yoco_extend_line_to_image_bounds(
    line: LineSegment,
    image_shape: Tuple[int, ...],
    is_vertical: bool
) -> Tuple[IntPoint, IntPoint]:
    """
    Étend une ligne jusqu'aux bords de l'image.
    
    Args:
        line: Segment de ligne
        image_shape: Dimensions de l'image
        is_vertical: True si la ligne est plutôt verticale
        
    Returns:
        Deux points définissant la ligne étendue
    """
    height, width = image_shape[:2]
    
    if is_vertical:
        # Étend vers le haut et le bas
        x0, y0 = line[0]
        x1, y1 = line[1]
        
        # Paramétrique: x = x0 + t*(x1-x0), y = y0 + t*(y1-y0)
        # Pour y = 0: t = -y0/(y1-y0)
        # Pour y = height: t = (height-y0)/(y1-y0)
        
        dy = y1 - y0
        if abs(dy) < 1e-8:
            return line[0], line[1]
        
        dx = x1 - x0
        
        t_top = -y0 / dy
        t_bottom = (height - y0) / dy
        
        pt_top = [int(x0 + t_top * dx), int(y0 + t_top * dy)]
        pt_bottom = [int(x0 + t_bottom * dx), int(y0 + t_bottom * dy)]
        
        return pt_top, pt_bottom
    else:
        # Étend vers la gauche et la droite
        x0, y0 = line[0]
        x1, y1 = line[1]
        
        dx = x1 - x0
        if abs(dx) < 1e-8:
            return line[0], line[1]
        
        dy = y1 - y0
        
        t_left = -x0 / dx
        t_right = (width - x0) / dx
        
        pt_left = [int(x0 + t_left * dx), int(y0 + t_left * dy)]
        pt_right = [int(x0 + t_right * dx), int(y0 + t_right * dy)]
        
        return pt_left, pt_right


def yoco_detect_inner_board_corners(
    image: np.ndarray,
    lattice_points: List[Point],
    lines: List[LineSegment]
) -> List[IntPoint]:
    """
    Détecte les quatre coins intérieurs de l'échiquier.
    
    Pipeline :
    1. Prépare et filtre les points du réseau
    2. Identifie les lignes candidates pour les bords
    3. Évalue tous les quadrilatères possibles
    4. Sélectionne le meilleur
    
    Args:
        image: Image de l'échiquier
        lattice_points: Points du réseau détectés
        lines: Lignes détectées
        
    Returns:
        Liste des 4 coins du plateau
        
    Example:
        >>> corners = yoco_detect_inner_board_corners(image, points, lines)
        >>> print(f"Coins: {corners}")
    """
    # Normalise et filtre les points
    points = yoco_normalize_points(lattice_points)
    points = yoco_filter_valid_points(points, image.shape)
    points = yoco_sort_points_clockwise(points)
    
    if len(points) < 4:
        raise ValueError("Pas assez de points pour détecter l'échiquier")
    
    # Paramètres
    area = cv2.contourArea(np.array(points))
    alpha = math.sqrt(area / 49)  # Taille moyenne d'une case
    beta = len(points) * (5 / 100)  # Tolérance
    
    # Filtre les points aberrants
    points = yoco_cluster_lattice_points(points, alpha)
    
    if len(points) < 4:
        raise ValueError("Pas assez de points après filtrage")
    
    # Calcule le centroïde
    centroid = (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points)
    )
    
    # Sépare les lignes verticales et horizontales
    vertical_lines = []
    horizontal_lines = []
    
    for line in lines:
        dx = abs(line[0][0] - line[1][0])
        dy = abs(line[0][1] - line[1][1])
        
        # Vérifie si la ligne passe près des points et loin du centre
        is_near_point = False
        is_far_from_center = True
        
        for pt in points:
            dist_to_line = yoco_calculate_point_to_line_distance(line, pt)
            if dist_to_line < alpha:
                is_near_point = True
                break
        
        dist_to_center = yoco_calculate_point_to_line_distance(line, list(centroid))
        if dist_to_center < alpha * 2.5:
            is_far_from_center = False
        
        if is_near_point and is_far_from_center:
            is_vertical = dx < dy
            extended = yoco_extend_line_to_image_bounds(line, image.shape, is_vertical)
            
            if is_vertical:
                vertical_lines.append(list(extended))
            else:
                horizontal_lines.append(list(extended))
    
    # Supprime les doublons
    vertical_lines = yoco_remove_duplicate_items(vertical_lines)
    horizontal_lines = yoco_remove_duplicate_items(horizontal_lines)
    
    if len(vertical_lines) < 2 or len(horizontal_lines) < 2:
        # Fallback: utilise l'enveloppe convexe
        hull = scipy.spatial.ConvexHull(np.array(points))
        hull_points = [points[i] for i in hull.vertices]
        approx = cv2.approxPolyDP(
            np.array(hull_points),
            0.01 * cv2.arcLength(np.array(hull_points), True),
            True
        )
        if len(approx) >= 4:
            return yoco_normalize_points([pt[0].tolist() for pt in approx[:4]])
        return yoco_sort_points_clockwise(hull_points[:4])
    
    # Évalue tous les quadrilatères candidats
    candidates: Dict[float, np.ndarray] = {}
    
    for v_pair in itertools.combinations(vertical_lines, 2):
        for h_pair in itertools.combinations(horizontal_lines, 2):
            # Trouve les 4 intersections
            all_lines = [v_pair[0], v_pair[1], h_pair[0], h_pair[1]]
            intersections = yoco_find_line_intersections(all_lines)
            intersections = yoco_filter_valid_points(
                yoco_normalize_points(intersections),
                image.shape
            )
            
            if len(intersections) != 4:
                continue
            
            # Trie et vérifie la convexité
            sorted_corners = np.array(yoco_sort_points_clockwise(intersections))
            
            if not cv2.isContourConvex(sorted_corners):
                continue
            
            # Calcule le score
            score = yoco_score_quadrilateral(
                sorted_corners, points, centroid,
                alpha=alpha / 2, beta=beta
            )
            
            if score > 0:
                candidates[-score] = sorted_corners  # Négatif pour tri décroissant
    
    if not candidates:
        raise ValueError("Aucun quadrilatère valide trouvé")
    
    # Sélectionne le meilleur
    best_corners = candidates[min(candidates.keys())]
    
    return yoco_normalize_points(best_corners.tolist())


def yoco_pad_board_corners(
    corners: List[IntPoint],
    image: np.ndarray,
    padding: int = 60
) -> List[IntPoint]:
    """
    Ajoute du padding autour des coins de l'échiquier.
    
    Étend le quadrilatère vers l'extérieur pour capturer
    les bords de l'échiquier qui pourraient être coupés.
    
    Args:
        corners: Quatre coins du plateau
        image: Image de référence
        padding: Quantité de padding en pixels
        
    Returns:
        Coins avec padding appliqué
    """
    if not PYCLIPPER_AVAILABLE:
        # Sans pyclipper, retourne les coins originaux
        return corners
    
    try:
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(corners, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        padded = pco.Execute(padding)
        
        if padded and len(padded[0]) >= 4:
            return padded[0]
    except:
        pass
    
    return corners