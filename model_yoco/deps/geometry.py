"""
Module de géométrie pour la détection d'intersections de segments.

Ce module fournit des fonctions pour calculer les intersections entre
segments de droites, utilisées pour détecter les points du réseau
de l'échiquier.
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

# Type aliases pour la lisibilité
Point = Tuple[float, float]
Segment = Tuple[Point, Point]


@dataclass
class IntersectionResult:
    """Résultat d'une intersection entre deux segments."""
    point: Point
    segment1_index: int
    segment2_index: int


def yoco_calculate_line_intersection(
    p1: Point, p2: Point, 
    p3: Point, p4: Point
) -> Optional[Point]:
    """
    Calcule le point d'intersection entre deux segments de droite.
    
    Utilise la méthode des déterminants pour trouver l'intersection
    entre le segment [p1, p2] et le segment [p3, p4].
    
    Args:
        p1, p2: Points définissant le premier segment
        p3, p4: Points définissant le second segment
        
    Returns:
        Le point d'intersection si les segments se croisent, None sinon
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Calcul du dénominateur
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    # Segments parallèles
    if abs(denominator) < 1e-10:
        return None
    
    # Calcul des paramètres t et u
    t_numerator = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    u_numerator = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))
    
    t = t_numerator / denominator
    u = u_numerator / denominator
    
    # Vérifie si l'intersection est sur les deux segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection_x = x1 + t * (x2 - x1)
        intersection_y = y1 + t * (y2 - y1)
        return (intersection_x, intersection_y)
    
    return None


def yoco_extend_segment(segment: Segment, extension_factor: float = 0.1) -> Segment:
    """
    Étend un segment dans les deux directions.
    
    Args:
        segment: Le segment à étendre
        extension_factor: Facteur d'extension (0.1 = 10% de chaque côté)
        
    Returns:
        Le segment étendu
    """
    (x1, y1), (x2, y2) = segment
    
    dx = x2 - x1
    dy = y2 - y1
    
    new_x1 = x1 - dx * extension_factor
    new_y1 = y1 - dy * extension_factor
    new_x2 = x2 + dx * extension_factor
    new_y2 = y2 + dy * extension_factor
    
    return ((new_x1, new_y1), (new_x2, new_y2))


def yoco_point_distance(p1: Point, p2: Point) -> float:
    """Calcule la distance euclidienne entre deux points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def yoco_is_point_near_segment_endpoint(
    point: Point, 
    segment: Segment, 
    threshold: float = 1e-6
) -> bool:
    """
    Vérifie si un point est proche d'une extrémité du segment.
    
    Args:
        point: Le point à vérifier
        segment: Le segment de référence
        threshold: Distance seuil
        
    Returns:
        True si le point est proche d'une extrémité
    """
    return (yoco_point_distance(point, segment[0]) < threshold or
            yoco_point_distance(point, segment[1]) < threshold)


def yoco_find_all_intersections_naive(segments: List[Segment]) -> List[Point]:
    """
    Trouve toutes les intersections entre segments (méthode naïve O(n²)).
    
    Cette méthode compare chaque paire de segments pour trouver
    les intersections. Simple mais efficace pour un nombre modéré
    de segments (< 1000).
    
    Args:
        segments: Liste des segments à analyser
        
    Returns:
        Liste des points d'intersection uniques
    """
    intersections: Set[Tuple[float, float]] = set()
    n = len(segments)
    
    for i in range(n):
        seg1 = segments[i]
        for j in range(i + 1, n):
            seg2 = segments[j]
            
            # Calcule l'intersection
            intersection = yoco_calculate_line_intersection(
                seg1[0], seg1[1],
                seg2[0], seg2[1]
            )
            
            if intersection is None:
                continue
            
            # Ignore les intersections aux extrémités
            if (yoco_is_point_near_segment_endpoint(intersection, seg1) and
                yoco_is_point_near_segment_endpoint(intersection, seg2)):
                continue
            
            # Arrondit pour éviter les doublons dus aux erreurs de précision
            rounded_point = (round(intersection[0], 6), round(intersection[1], 6))
            intersections.add(rounded_point)
    
    return list(intersections)


def yoco_find_intersections_sweep_line(segments: List[Segment]) -> List[Point]:
    """
    Trouve les intersections avec l'algorithme de balayage (sweep line).
    
    Implémentation simplifiée de l'algorithme de Bentley-Ottmann.
    Plus efficace que la méthode naïve pour un grand nombre de segments.
    
    Args:
        segments: Liste des segments à analyser
        
    Returns:
        Liste des points d'intersection
    """
    if len(segments) < 50:
        # Pour peu de segments, la méthode naïve est plus simple
        return yoco_find_all_intersections_naive(segments)
    
    # Normalise les segments (point gauche en premier)
    normalized_segments = []
    for seg in segments:
        if seg[0][0] > seg[1][0] or (seg[0][0] == seg[1][0] and seg[0][1] > seg[1][1]):
            normalized_segments.append((seg[1], seg[0]))
        else:
            normalized_segments.append(seg)
    
    # Événements triés par coordonnée x
    events = []
    for idx, seg in enumerate(normalized_segments):
        events.append((seg[0][0], 'start', idx, seg))
        events.append((seg[1][0], 'end', idx, seg))
    
    events.sort(key=lambda e: (e[0], 0 if e[1] == 'start' else 1))
    
    # Segments actifs
    active_segments = set()
    intersections = set()
    
    for event in events:
        x_coord, event_type, seg_idx, segment = event
        
        if event_type == 'start':
            # Vérifie les intersections avec tous les segments actifs
            for active_idx in active_segments:
                active_seg = normalized_segments[active_idx]
                intersection = yoco_calculate_line_intersection(
                    segment[0], segment[1],
                    active_seg[0], active_seg[1]
                )
                
                if intersection is not None:
                    # Ignore les intersections aux extrémités
                    if not (yoco_is_point_near_segment_endpoint(intersection, segment) and
                            yoco_is_point_near_segment_endpoint(intersection, active_seg)):
                        rounded = (round(intersection[0], 6), round(intersection[1], 6))
                        intersections.add(rounded)
            
            active_segments.add(seg_idx)
        else:
            active_segments.discard(seg_idx)
    
    return list(intersections)


def isect_segments(segments: List[Segment]) -> List[Point]:
    """
    Interface principale pour trouver les intersections de segments.
    
    Cette fonction est le point d'entrée principal du module.
    Elle choisit automatiquement l'algorithme le plus adapté
    selon le nombre de segments.
    
    Args:
        segments: Liste de segments, chaque segment étant un tuple
                 de deux points ((x1, y1), (x2, y2))
                 
    Returns:
        Liste des points d'intersection
        
    Example:
        >>> segments = [((0, 0), (10, 10)), ((0, 10), (10, 0))]
        >>> intersections = isect_segments(segments)
        >>> print(intersections)  # [(5.0, 5.0)]
    """
    if not segments:
        return []
    
    # Convertit en tuples si nécessaire
    converted_segments = []
    for seg in segments:
        p1 = (float(seg[0][0]), float(seg[0][1]))
        p2 = (float(seg[1][0]), float(seg[1][1]))
        converted_segments.append((p1, p2))
    
    return yoco_find_intersections_sweep_line(converted_segments)