"""
Utilitaires pour les transformations d'images d'échiquiers.
Contient les fonctions de redimensionnement et de transformation de perspective.
"""
import functools
import numpy as np
import cv2
import math
import sys
import os
# Ajoute le répertoire src au path pour les imports relatifs
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, src_dir)

from constant import RESIZE_HEIGHT, SQUARE_LENGTH

_np_array = np.array


def yoco_scale_coordinates(coordinate_points, scale_factor):
    """
    Redimensionne des points de coordonnées selon un facteur d'échelle.
    
    Args:
        coordinate_points: Liste de points à redimensionner
        scale_factor: Facteur d'échelle à appliquer
        
    Returns:
        Liste de points redimensionnés
    """
    def scale_single_point(point, scale):
        return [point[0] * scale, point[1] * scale]
    
    return list(map(functools.partial(scale_single_point, scale=1/scale_factor), coordinate_points))


def yoco_resize_image_constant_area(image_array, target_height=RESIZE_HEIGHT):
    """
    Redimensionne une image en conservant la même surface (surface = height²).
    
    Args:
        image_array: Image à redimensionner
        target_height: Hauteur de référence pour le calcul de la surface
        
    Returns:
        Tuple (image_redimensionnée, forme_de_l_image, facteur_d_échelle)
    """
    target_pixel_count = target_height * target_height
    image_shape = list(np.shape(image_array))
    calculated_scale = math.sqrt(float(target_pixel_count) / float(image_shape[0] * image_shape[1]))
    
    image_shape[0] *= calculated_scale
    image_shape[1] *= calculated_scale
    resized_image_array = cv2.resize(image_array, (int(image_shape[1]), int(image_shape[0])))
    output_shape = np.shape(resized_image_array)
    
    return resized_image_array, output_shape, calculated_scale


def yoco_apply_perspective_transform(image_array, corner_points, chess_square_length=SQUARE_LENGTH):
    """
    Applique une transformation de perspective pour recadrer l'échiquier.
    
    Args:
        image_array: Image source
        corner_points: Quatre points définissant les coins de l'échiquier
        chess_square_length: Longueur d'une case d'échecs en pixels
        
    Returns:
        Image transformée avec l'échiquier rectifié
    """
    chessboard_edge_length = chess_square_length * 8
    
    def calculate_point_distance(point_a, point_b):
        """Calcule la distance euclidienne entre deux points."""
        return np.linalg.norm(_np_array(point_a) - _np_array(point_b))
    
    def circular_shift_sequence(sequence, shift_amount=0):
        """Décale une séquence circulairement de n positions."""
        return sequence[-(shift_amount % len(sequence)):] + sequence[:-(shift_amount % len(sequence))]
    
    # Trouve le point le plus proche de l'origine (coin supérieur gauche)
    best_point_index, minimum_distance = 0, 10**6
    for point_index, point_coord in enumerate(corner_points):
        distance_to_origin = calculate_point_distance(point_coord, [0, 0])
        if distance_to_origin < minimum_distance:
            best_point_index, minimum_distance = point_index, distance_to_origin
    
    # Réorganise les points pour commencer par le coin supérieur gauche
    reordered_source_points = np.float32(circular_shift_sequence(corner_points, 4 - best_point_index))
    
    # Points de destination pour un échiquier parfaitement carré
    target_destination_points = np.float32([
        [0, 0],
        [chessboard_edge_length, 0],
        [chessboard_edge_length, chessboard_edge_length],
        [0, chessboard_edge_length]
    ])
    
    # Calcule la matrice de transformation et applique la transformation
    perspective_matrix = cv2.getPerspectiveTransform(reordered_source_points, target_destination_points)
    transformed_image_array = cv2.warpPerspective(image_array, perspective_matrix, (chessboard_edge_length, chessboard_edge_length))
    
    return transformed_image_array


def yoco_crop_chessboard_image(image_array, corner_points, scale_factor):
    """
    Recadre une image en appliquant une transformation de perspective.
    
    Args:
        image_array: Image à recadrer
        corner_points: Points définissant la zone à recadrer (dans l'espace redimensionné)
        scale_factor: Facteur d'échelle pour convertir les points vers l'espace original
        
    Returns:
        Image recadrée
    """
    # Convertit les points vers l'espace original
    original_scale_points = yoco_scale_coordinates(corner_points, scale_factor)
    # Applique la transformation de perspective
    cropped_image_result = yoco_apply_perspective_transform(image_array, original_scale_points)
    return cropped_image_result

