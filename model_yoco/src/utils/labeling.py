"""
Utilitaires pour la création de labels à partir de fichiers PGN et d'images préprocessées.
"""
import numpy as np
from matplotlib import pyplot as plt
import cv2
import chess.pgn
from pathlib import Path
import sys
import os
# Ajoute le répertoire src au path pour les imports relatifs
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, src_dir)

from constant import (
    PGN_FOLDER, LABELED_DATA_FOLDER, PREPROCESSED_DATA_FOLDER, 
    PIECE_LABELS, image_counter
)


def yoco_create_labels_from_pgn(game_name, is_reversed_perspective=False):
    """
    Labellise une partie d'échecs en extrayant chaque case de l'échiquier
    et en l'associant à la pièce correspondante depuis le fichier PGN.
    
    Args:
        game_name: Nom de la partie à labelliser
        is_reversed_perspective: Si True, traite l'image depuis la perspective noire
    """
    global image_counter
    
    # Ouvre le fichier PGN de la partie
    pgn_file_path = PGN_FOLDER + '%s.pgn' % game_name
    with open(pgn_file_path) as pgn_file:
        chess_game = chess.pgn.read_game(pgn_file)
    chess_board = chess_game.board()
    
    move_counter = 0
    for chess_move in chess_game.mainline_moves():
        chess_board.push(chess_move)
        move_counter += 1
        
        # Détermine le chemin de l'image selon la perspective
        perspective_version = 'rev' if is_reversed_perspective else 'orig'
        chess_image_path = PREPROCESSED_DATA_FOLDER + '%s/%s/%i.png' % (game_name, perspective_version, move_counter)
        chess_image = cv2.imread(chess_image_path)
        
        if chess_image is None:
            return
        
        image_height_dim, image_width_dim, _ = chess_image.shape
        
        # Calcule la taille d'une case d'échecs
        square_height_dim = image_height_dim // 8
        square_width_dim = image_width_dim // 8
        square_position_index = 0
        
        if not is_reversed_perspective:
            # Traitement depuis la perspective blanche (haut vers bas, gauche vers droite)
            for y_coord in range(image_height_dim-1, -1, -square_height_dim):
                for x_coord in range(0, image_width_dim, square_width_dim):
                    chess_piece = str(chess_board.piece_at(square_position_index))
                    square_image = chess_image[max(0, y_coord-2*square_height_dim):y_coord, x_coord:x_coord+square_width_dim]
                    
                    # Ajoute du padding si nécessaire
                    if y_coord-2*square_height_dim < 0:
                        padding_array = np.zeros((2*square_height_dim-y_coord, square_width_dim, 3))
                        square_image = np.concatenate((padding_array, square_image))
                        square_image = square_image.astype(np.uint8)
                    
                    # Détermine le dossier de destination selon la pièce
                    if chess_piece == 'None':
                        output_directory_path = LABELED_DATA_FOLDER + 'Empty/'
                    else:
                        piece_type_name = PIECE_LABELS[chess_piece.lower()]
                        piece_color_name = 'White' if chess_piece.isupper() else 'Black'
                        output_directory_path = LABELED_DATA_FOLDER + '%s_%s/' % (piece_type_name, piece_color_name)
                    
                    Path(output_directory_path).mkdir(parents=True, exist_ok=True)
                    image_counter += 1
                    square_position_index += 1
                    plt.imsave(output_directory_path + '%i.jpg' % image_counter, square_image)
        else:
            # Traitement depuis la perspective noire (bas vers haut, droite vers gauche)
            for y_coord in range(square_height_dim, image_height_dim+1, square_height_dim):
                for x_coord in range(image_width_dim, square_width_dim-1, -square_width_dim):
                    chess_piece = str(chess_board.piece_at(square_position_index))
                    square_image = chess_image[max(0, y_coord-2*square_height_dim):y_coord, x_coord-square_width_dim:x_coord]
                    
                    # Ajoute du padding si nécessaire
                    if y_coord-2*square_height_dim < 0:
                        padding_array = np.zeros((2*square_height_dim-y_coord, square_width_dim, 3))
                        square_image = np.concatenate((padding_array, square_image))
                        square_image = square_image.astype(np.uint8)
                    
                    # Détermine le dossier de destination selon la pièce
                    if chess_piece == 'None':
                        output_directory_path = LABELED_DATA_FOLDER + 'Empty/'
                    else:
                        piece_type_name = PIECE_LABELS[chess_piece.lower()]
                        piece_color_name = 'White' if chess_piece.isupper() else 'Black'
                        output_directory_path = LABELED_DATA_FOLDER + '%s_%s/' % (piece_type_name, piece_color_name)
                    
                    Path(output_directory_path).mkdir(parents=True, exist_ok=True)
                    image_counter += 1
                    square_position_index += 1
                    plt.imsave(output_directory_path + '%i.jpg' % image_counter, square_image)


if __name__ == '__main__':
    # Liste de toutes les parties photographiées présentes dans notre dataset
    chess_games_list = [
        'carlsen_anand_2014', 
        'carlsen_gukesh_2025', 
        'david_vachier-lagrave_2014', 
        'nataf_vachier-lagrave_2006', 
        'vachier-lagrave_carlsen_2023'
    ]
    
    # Traite chaque partie depuis les deux perspectives (blanche et noire)
    for game_name_item in chess_games_list:
        yoco_create_labels_from_pgn(game_name_item, is_reversed_perspective=False)  # Perspective blanche
        yoco_create_labels_from_pgn(game_name_item, is_reversed_perspective=True)   # Perspective noire

