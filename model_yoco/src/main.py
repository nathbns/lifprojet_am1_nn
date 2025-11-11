"""
Script principal pour analyser des images d'échiquiers et prédire les positions.
Usage: python src/main.py chemin/vers/image.jpg
"""
import chess
import chess.svg
import io
import json
import numpy as np
import os
import sys
import wand.color
import wand.image
from matplotlib import pyplot as plt
from pathlib import Path
from wand.api import library

# Ajoute les répertoires nécessaires au path pour les imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)  # Pour importer deps
sys.path.insert(0, current_dir)   # Pour les imports relatifs dans src

from preprocessing.image_processing import yoco_preprocess_chessboard_image
from training.model_training import yoco_create_cnn_model
from constant import (
    CLASS_INDICES_FILE, FEN_LABELS, PIECE_CLASSES,
    TEMP_SVG_FOLDER, RESULTS_FOLDER, MODEL_WEIGHTS_FILE, IMAGE_SIZE
)

# Charge l'ordre des classes depuis le fichier généré par train.py
piece_classes_ordered_list = PIECE_CLASSES
if os.path.exists(CLASS_INDICES_FILE):
    try:
        with open(CLASS_INDICES_FILE, 'r') as f:
            class_indices_dict = json.load(f)
        # Inverse pour avoir index -> nom
        piece_classes_ordered_list = [None] * len(class_indices_dict)
        for class_name, class_index in class_indices_dict.items():
            piece_classes_ordered_list[class_index] = class_name
        print(f"Ordre des classes chargé: {piece_classes_ordered_list}")
    except Exception as e:
        print(f"Erreur lors du chargement de {CLASS_INDICES_FILE}: {e}")
        print(f"Utilisation de l'ordre par défaut: {PIECE_CLASSES}")
else:
    print(f"Fichier {CLASS_INDICES_FILE} non trouvé, utilisation de l'ordre par défaut: {PIECE_CLASSES}")


def yoco_classify_chess_piece_image(piece_image_array, cnn_model):
    """
    Classe une image de pièce d'échecs dans l'une des classes définies.
    
    Args:
        piece_image_array: Image d'une pièce d'échecs (tableau numpy)
        cnn_model: Modèle de classification entraîné
        
    Returns:
        Nom de la classe prédite (ex: 'King_White', 'Pawn_Black', etc.)
    """
    height, width = IMAGE_SIZE
    # Normalise l'image comme lors de l'entraînement (division par 255)
    if piece_image_array.max() > 1.0:
        piece_image_array = piece_image_array.astype(np.float32) / 255.0
    else:
        piece_image_array = piece_image_array.astype(np.float32)
    
    predictions_result = cnn_model.predict(piece_image_array.reshape(1, height, width, 3), verbose=0)
    predicted_class_index = predictions_result.argmax()
    return piece_classes_ordered_list[predicted_class_index]


def yoco_analyze_chessboard_position(board_image_array, cnn_model):
    """
    Analyse une image d'échiquier complet et retourne un tableau représentant
    la position d'échecs prédite. Note: la première ligne du tableau correspond
    à la 8ème rangée de l'échiquier (où se trouvent toutes les pièces noires
    non-pions au départ).
    
    Args:
        board_image_array: Image complète d'un échiquier
        cnn_model: Modèle de classification entraîné
        
    Returns:
        Tableau 8x8 représentant la position d'échecs en notation FEN
    """
    board_position = []
    square_height_dim = board_image_array.shape[0] // 8
    square_width_dim = board_image_array.shape[1] // 8
    
    # Parcourt l'échiquier case par case
    for y_coordinate in range(square_height_dim-1, board_image_array.shape[1], square_height_dim):
        row_position = []
        for x_coordinate in range(0, board_image_array.shape[1], square_width_dim):
            square_image_patch = board_image_array[max(0, y_coordinate-2*square_height_dim):y_coordinate, x_coordinate:x_coordinate+square_width_dim]
            
            # Ajoute du padding si nécessaire
            if y_coordinate-2*square_height_dim < 0:
                padding_array = np.zeros((2*square_height_dim-y_coordinate, square_width_dim, 3))
                square_image_patch = np.concatenate((padding_array, square_image_patch))
                square_image_patch = square_image_patch.astype(np.uint8)

            piece_class_prediction = yoco_classify_chess_piece_image(square_image_patch, cnn_model)
            row_position.append(FEN_LABELS[piece_class_prediction])
        board_position.append(row_position)

    # Vérifie et corrige les rois manquants (parfois confondus avec les dames)
    has_white_king_flag = False
    has_black_king_flag = False
    white_queen_position = (-1, -1)
    black_queen_position = (-1, -1)
    
    for row_index in range(8):
        for col_index in range(8):
            if board_position[row_index][col_index] == 'K':
                has_white_king_flag = True
            if board_position[row_index][col_index] == 'k':
                has_black_king_flag = True
            if board_position[row_index][col_index] == 'Q':
                white_queen_position = (row_index, col_index)
            if board_position[row_index][col_index] == 'q':
                black_queen_position = (row_index, col_index)
    
    # Si un roi manque mais qu'une dame est présente, remplace la dame par le roi
    if not has_white_king_flag and white_queen_position[0] >= 0:
        board_position[white_queen_position[0]][white_queen_position[1]] = 'K'
    if not has_black_king_flag and black_queen_position[0] >= 0:
        board_position[black_queen_position[0]][black_queen_position[1]] = 'k'

    return board_position


def yoco_convert_board_to_fen(board_position_array):
    """
    Convertit un tableau représentant une position d'échecs en notation FEN.
    
    Args:
        board_position_array: Tableau 8x8 représentant la position
        
    Returns:
        Chaîne de caractères au format FEN
    """
    with io.StringIO() as fen_string_buffer:
        for row in board_position_array:
            empty_squares_count = 0
            for cell in row:
                if cell != '.':
                    if empty_squares_count > 0:
                        fen_string_buffer.write(str(empty_squares_count))
                        empty_squares_count = 0
                    fen_string_buffer.write(cell)
                else:
                    empty_squares_count += 1
            if empty_squares_count > 0:
                fen_string_buffer.write(str(empty_squares_count))
            fen_string_buffer.write('/')
        
        # Supprime le dernier '/' et ajoute les informations de la partie
        fen_string_buffer.seek(fen_string_buffer.tell() - 1)
        fen_string_buffer.write(' w KQkq - 0 1')
        return fen_string_buffer.getvalue()


def yoco_convert_fen_to_svg(fen_string):
    """
    Convertit une notation FEN en image SVG de l'échiquier.
    
    Args:
        fen_string: Notation FEN de la position d'échecs
    """
    chess_board = chess.Board(fen_string)
    board_svg_result = chess.svg.board(board=chess_board)
    temp_svg_path = os.path.join(TEMP_SVG_FOLDER, 'temp.SVG')
    with open(temp_svg_path, "w") as svg_output_file:
        svg_output_file.write(board_svg_result)


def yoco_convert_svg_to_png(svg_input_file, png_output_file, dpi_resolution=300):
    """
    Convertit un fichier SVG en image PNG.
    
    Args:
        svg_input_file: Chemin vers le fichier SVG
        png_output_file: Chemin vers le fichier PNG de sortie
        dpi_resolution: Résolution de l'image (points par pouce)
    """
    with wand.image.Image(resolution=dpi_resolution) as image_wand:
        with wand.color.Color('transparent') as background_color:
            library.MagickSetBackgroundColor(image_wand.wand,
                                             background_color.resource)
        image_wand.read(filename=svg_input_file, resolution=dpi_resolution)
        png_image_blob = image_wand.make_blob("png32")
        with open(png_output_file, "wb") as output_file:
            output_file.write(png_image_blob)
        os.remove(svg_input_file)


if __name__ == '__main__':
    # Vérifie que le fichier de poids du modèle existe
    if not os.path.exists(MODEL_WEIGHTS_FILE):
        print(f"ERREUR: Le fichier de poids du modèle n'existe pas: {MODEL_WEIGHTS_FILE}")
        print(f"Veuillez vous assurer que le modèle a été entraîné et que les poids sont sauvegardés.")
        sys.exit(1)
    
    # Traite chaque image fournie en argument
    for input_image_path in sys.argv[1:]:
        # Convertit le chemin en absolu si nécessaire
        if not os.path.isabs(input_image_path):
            # Si le chemin est relatif, on le résout depuis le répertoire de travail actuel
            input_image_path = os.path.abspath(input_image_path)
        
        if not os.path.exists(input_image_path):
            print(f"ERREUR: Le fichier image n'existe pas: {input_image_path}")
            continue
            
        image_filename = os.path.basename(input_image_path)
        output_image_file = os.path.join(RESULTS_FOLDER, image_filename)
        Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)
        
        print(f"Chargement du modèle depuis: {MODEL_WEIGHTS_FILE}")
        # Charge le modèle et ses poids
        cnn_model = yoco_create_cnn_model()
        cnn_model.load_weights(MODEL_WEIGHTS_FILE)
        print("Modèle chargé avec succès!")
        
        # Prétraite l'image, analyse la position et génère la visualisation
        print(f"Traitement de l'image: {input_image_path}")
        processed_image_result = yoco_preprocess_chessboard_image(input_image_path, should_save=False)
        board_position_result = yoco_analyze_chessboard_position(processed_image_result, cnn_model)
        fen_string_result = yoco_convert_board_to_fen(board_position_result)
        print(f"Position FEN détectée: {fen_string_result}")
        yoco_convert_fen_to_svg(fen_string_result)
        temp_svg_path = os.path.join(TEMP_SVG_FOLDER, 'temp.SVG')
        yoco_convert_svg_to_png(svg_input_file=temp_svg_path,
                   png_output_file=output_image_file)
        print(f"Résultat sauvegardé dans: {output_image_file}")
    
    print('Terminé!')

