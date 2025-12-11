"""
Module de détection des lignes de l'échiquier.

Ce module implémente l'algorithme SLID (Straight Line Detector) adapté
pour la détection des lignes d'un échiquier dans une image.

L'approche utilise :
1. Amélioration du contraste avec CLAHE (Contrast Limited Adaptive Histogram Equalization)
2. Détection de contours avec Canny
3. Détection de lignes avec la transformée de Hough probabiliste
4. Regroupement des lignes similaires avec Union-Find
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass

Point = Tuple[int, int]
LineSegment = List[Point]  # [[x1, y1], [x2, y2]]


@dataclass
class CLAHEConfig:
    """Configuration pour l'algorithme CLAHE."""
    clip_limit: int
    grid_size: Tuple[int, int]
    iterations: int


# Configurations CLAHE optimisées pour la détection d'échiquiers
# Chaque configuration capture différents aspects des lignes
YOCO_CLAHE_CONFIGS = [
    CLAHEConfig(clip_limit=3, grid_size=(2, 6), iterations=5),  # Lignes horizontales
    CLAHEConfig(clip_limit=3, grid_size=(6, 2), iterations=5),  # Lignes verticales
    CLAHEConfig(clip_limit=5, grid_size=(3, 3), iterations=5),  # Équilibré
    CLAHEConfig(clip_limit=0, grid_size=(0, 0), iterations=0),  # Sans amélioration
]


def yoco_apply_clahe_enhancement(
    image: np.ndarray,
    config: CLAHEConfig
) -> np.ndarray:
    """
    Applique l'amélioration de contraste CLAHE à une image.
    
    CLAHE améliore le contraste local de l'image, ce qui aide
    à mieux détecter les lignes de l'échiquier.
    
    Args:
        image: Image BGR en entrée
        config: Configuration CLAHE à appliquer
        
    Returns:
        Image en niveaux de gris avec contraste amélioré
    """
    # Convertit en niveaux de gris
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Cas sans amélioration
    if config.clip_limit == 0:
        return gray_image
    
    # Applique CLAHE plusieurs fois pour renforcer le contraste
    enhanced_image = gray_image.copy()
    for _ in range(config.iterations):
        clahe = cv2.createCLAHE(
            clipLimit=config.clip_limit,
            tileGridSize=config.grid_size
        )
        enhanced_image = clahe.apply(enhanced_image)
    
    # Applique une fermeture morphologique pour connecter les lignes
        kernel = np.ones((10, 10), np.uint8)
    enhanced_image = cv2.morphologyEx(enhanced_image, cv2.MORPH_CLOSE, kernel)
    
    return enhanced_image


def yoco_detect_edges_canny(
    image: np.ndarray,
    sigma: float = 0.25
) -> np.ndarray:
    """
    Détecte les contours avec l'algorithme de Canny.
    
    Utilise des seuils automatiques basés sur la médiane
    de l'image pour s'adapter à différentes conditions d'éclairage.
    
    Args:
        image: Image en niveaux de gris
        sigma: Paramètre pour calculer les seuils (0.25 par défaut)
        
    Returns:
        Image binaire des contours
    """
    # Calcule les seuils basés sur la médiane
    median_value = np.median(image)
    lower_threshold = int(max(0, (1.0 - sigma) * median_value))
    upper_threshold = int(min(255, (1.0 + sigma) * median_value))
    
    # Applique un flou pour réduire le bruit
    blurred = cv2.medianBlur(image, 5)
    blurred = cv2.GaussianBlur(blurred, (7, 7), 2)
    
    # Détecte les contours
    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
    
    return edges


def yoco_detect_lines_hough(
    edge_image: np.ndarray,
    threshold: int = 40,
    min_line_length: int = 50,
    max_line_gap: int = 15
) -> List[LineSegment]:
    """
    Détecte les lignes avec la transformée de Hough probabiliste.
    
    Args:
        edge_image: Image binaire des contours
        threshold: Seuil minimum de votes
        min_line_length: Longueur minimale des lignes
        max_line_gap: Écart maximum entre segments d'une même ligne
        
    Returns:
        Liste des segments de lignes détectés
    """
    lines = cv2.HoughLinesP(
        edge_image,
        rho=1,
        theta=np.pi / 180,  # Résolution angulaire de 1 degré
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    if lines is None:
        return []
    
    # Convertit au format [[x1, y1], [x2, y2]]
    detected_lines = []
    for line in lines.reshape(-1, 4):
        segment = [[int(line[0]), int(line[1])], [int(line[2]), int(line[3])]]
        detected_lines.append(segment)
    
    return detected_lines


def yoco_collect_line_segments(image: np.ndarray) -> List[LineSegment]:
    """
    Collecte tous les segments de lignes avec différentes configurations CLAHE.
    
    Applique plusieurs configurations pour capturer le maximum de lignes
    dans différentes conditions.
    
    Args:
        image: Image BGR de l'échiquier
        
    Returns:
        Liste de tous les segments détectés
    """
    all_segments = []
    
    for config in YOCO_CLAHE_CONFIGS:
        # Améliore le contraste
        enhanced = yoco_apply_clahe_enhancement(image, config)
        
        # Détecte les contours
        edges = yoco_detect_edges_canny(enhanced)
        
        # Détecte les lignes
        segments = yoco_detect_lines_hough(edges)
        all_segments.extend(segments)
    
    return all_segments


class UnionFind:
    """
    Structure Union-Find pour regrouper les lignes similaires.
    
    Permet de fusionner efficacement des ensembles de lignes
    qui représentent la même ligne de l'échiquier.
    """
    
    def __init__(self):
        self.parent: Dict[int, int] = {}
        self.groups: Dict[int, Set[int]] = {}
    
    def find(self, x: int) -> int:
        """Trouve le représentant du groupe de x."""
        if x not in self.parent:
            self.parent[x] = x
            self.groups[x] = {x}
        
        # Compression de chemin
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self, a: int, b: int) -> None:
        """Fusionne les groupes de a et b."""
        root_a = self.find(a)
        root_b = self.find(b)
        
        if root_a != root_b:
            # Fusionne le plus petit groupe dans le plus grand
            if len(self.groups[root_a]) < len(self.groups[root_b]):
                root_a, root_b = root_b, root_a
            
            self.parent[root_b] = root_a
            self.groups[root_a] |= self.groups[root_b]


def yoco_calculate_segment_distance(p1: Point, p2: Point) -> float:
    """Calcule la distance euclidienne entre deux points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def yoco_calculate_point_to_line_distance(
    line: LineSegment,
    point: Point
) -> float:
    """
    Calcule la distance perpendiculaire d'un point à une ligne.
    
    Args:
        line: Segment de ligne [[x1, y1], [x2, y2]]
        point: Point [x, y]
        
    Returns:
        Distance perpendiculaire
    """
    line_vec = np.array(line[1]) - np.array(line[0])
    point_vec = np.array(point) - np.array(line[0])
    
    line_length = np.linalg.norm(line_vec)
    if line_length < 1e-8:
        return np.linalg.norm(point_vec)
    
    cross_product = np.abs(np.cross(line_vec, point_vec))
    return cross_product / line_length


def yoco_are_lines_similar(line1: LineSegment, line2: LineSegment) -> bool:
    """
    Détermine si deux segments représentent la même ligne.
    
    Deux lignes sont considérées similaires si :
    - Elles ont une orientation proche
    - La distance entre elles est faible par rapport à leur longueur
    
    Args:
        line1, line2: Segments à comparer
        
    Returns:
        True si les lignes sont similaires
    """
    # Longueurs des segments
    len1 = yoco_calculate_segment_distance(line1[0], line1[1])
    len2 = yoco_calculate_segment_distance(line2[0], line2[1])
    
    # Distances perpendiculaires
    d1a = yoco_calculate_point_to_line_distance(line2, line1[0])
    d1b = yoco_calculate_point_to_line_distance(line2, line1[1])
    d2a = yoco_calculate_point_to_line_distance(line1, line2[0])
    d2b = yoco_calculate_point_to_line_distance(line1, line2[1])
    
    # Distance moyenne
    avg_distance = 0.25 * (d1a + d1b + d2a + d2b)
    
    # Cas où les lignes sont presque identiques
    if avg_distance < 1e-8:
        return True
    
    # Seuil basé sur la longueur des lignes
    threshold = 0.0625 * (len1 + len2)
    
    # Les lignes sont similaires si le ratio longueur/distance est suffisant
    return (len1 / avg_distance > threshold) and (len2 / avg_distance > threshold)


def yoco_generate_points_on_segment(
    start: Point,
    end: Point,
    num_points: int = 10
) -> List[Point]:
    """Génère des points uniformément répartis sur un segment."""
    points = []
    for i in range(num_points):
        t = i / num_points
        x = int(start[0] + (end[0] - start[0]) * t)
        y = int(start[1] + (end[1] - start[1]) * t)
        points.append([x, y])
    return points


def yoco_fit_line_to_group(
    segments: List[LineSegment],
    segment_map: Dict[int, LineSegment]
) -> LineSegment:
    """
    Ajuste une ligne unique à un groupe de segments.
    
    Utilise cv2.fitLine pour trouver la meilleure ligne
    passant par tous les points des segments du groupe.
    
    Args:
        segments: Indices des segments du groupe
        segment_map: Mapping index -> segment
        
    Returns:
        Segment de ligne ajusté
    """
    # Collecte tous les points
    all_points = []
    for seg_id in segments:
        segment = segment_map[seg_id]
        all_points.extend(yoco_generate_points_on_segment(segment[0], segment[1]))
    
    if not all_points:
        return [[0, 0], [0, 0]]
    
    points_array = np.array(all_points)
    
    # Calcule le rayon englobant pour déterminer la longueur de la ligne
    _, radius = cv2.minEnclosingCircle(points_array)
    half_length = radius * np.pi / 2
    
    # Ajuste la ligne
    vx, vy, cx, cy = cv2.fitLine(points_array, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # Crée le segment final
    x1 = int(cx - vx * half_length)
    y1 = int(cy - vy * half_length)
    x2 = int(cx + vx * half_length)
    y2 = int(cy + vy * half_length)
    
    return [[x1, y1], [x2, y2]]


def yoco_group_similar_lines(segments: List[LineSegment]) -> List[LineSegment]:
    """
    Regroupe les segments similaires et fusionne chaque groupe.
    
    Args:
        segments: Liste des segments bruts
        
    Returns:
        Liste des lignes fusionnées
    """
    if not segments:
        return []
    
    # Sépare les lignes verticales et horizontales
    vertical_lines = []
    horizontal_lines = []
    
    segment_map: Dict[int, LineSegment] = {}
    
    for idx, segment in enumerate(segments):
        segment_map[idx] = segment
        
        dx = abs(segment[0][0] - segment[1][0])
        dy = abs(segment[0][1] - segment[1][1])
        
        if dx < dy:
            vertical_lines.append(idx)
        else:
            horizontal_lines.append(idx)
    
    # Regroupe avec Union-Find
    uf = UnionFind()
    
    for line_group in [vertical_lines, horizontal_lines]:
        for i in range(len(line_group)):
            idx_i = line_group[i]
            uf.find(idx_i)  # Initialise
            
            for j in range(i + 1, len(line_group)):
                idx_j = line_group[j]
                
                if yoco_are_lines_similar(segment_map[idx_i], segment_map[idx_j]):
                    uf.union(idx_i, idx_j)
    
    # Crée les lignes finales
    final_lines = []
    processed_roots = set()
    
    for idx in segment_map.keys():
        root = uf.find(idx)
        if root not in processed_roots:
            processed_roots.add(root)
            group_segments = uf.groups[root]
            fitted_line = yoco_fit_line_to_group(group_segments, segment_map)
            final_lines.append(fitted_line)
    
    return final_lines


def yoco_extend_lines(lines: List[LineSegment], scale: float = 4) -> List[LineSegment]:
    """
    Étend les lignes pour qu'elles dépassent les bords de l'échiquier.
    
    Cela aide à s'assurer que les intersections aux bords sont détectées.
    
    Args:
        lines: Liste des lignes à étendre
        scale: Facteur d'extension
        
    Returns:
        Liste des lignes étendues
    """
    extended_lines = []
    
    def scale_point(x, y, s):
        return int(x * (1 + s) / 2 + y * (1 - s) / 2)
    
    for line in lines:
        a, b = line[0].copy(), line[1].copy()
        
        # Étend dans les deux directions
        new_a = [scale_point(a[0], b[0], scale), scale_point(a[1], b[1], scale)]
        new_b = [scale_point(b[0], a[0], scale), scale_point(b[1], a[1], scale)]
        
        extended_lines.append([new_a, new_b])
    
    return extended_lines


def yoco_detect_chessboard_lines(image: np.ndarray) -> List[LineSegment]:
    """
    Détecte les lignes de l'échiquier dans une image.
    
    Pipeline complet de détection :
    1. Collecte des segments avec différentes configurations CLAHE
    2. Regroupement des segments similaires
    3. Extension des lignes
    
    Args:
        image: Image BGR de l'échiquier
        
    Returns:
        Liste des lignes détectées [[x1, y1], [x2, y2]]
        
    Example:
        >>> image = cv2.imread("chessboard.jpg")
        >>> lines = yoco_detect_chessboard_lines(image)
        >>> print(f"Détecté {len(lines)} lignes")
    """
    # Étape 1: Collecte tous les segments
    segments = yoco_collect_line_segments(image)
    
    # Étape 2: Regroupe les segments similaires
    grouped_lines = yoco_group_similar_lines(segments)
    
    # Étape 3: Étend les lignes
    extended_lines = yoco_extend_lines(grouped_lines)
    
    return extended_lines