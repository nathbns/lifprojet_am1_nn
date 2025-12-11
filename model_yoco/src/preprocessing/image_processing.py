"""
Module de preprocessing des images d'échiquiers.
Contient les fonctions pour prétraiter les images et détecter/recadrer les échiquiers.
"""
from matplotlib import pyplot as plt
import glob
import cv2
from pathlib import Path
import sys
import os
# Ajoute le répertoire src au path pour les imports relatifs
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from detection.line_detection import yoco_detect_chessboard_lines
from detection.lattice_detection import yoco_detect_lattice_points
from detection.board_corners import yoco_detect_inner_board_corners, yoco_pad_board_corners
from utils.image_transforms import (
    yoco_resize_image_constant_area, yoco_crop_chessboard_image
)
from constant import RAW_DATA_FOLDER, PREPROCESSED_DATA_FOLDER


def yoco_preprocess_chessboard_image(image_file_path, output_directory="", output_filename="", should_save=False):
    """
    Prétraite une image d'échiquier en détectant et en recadrant le plateau.
    
    Args:
        image_file_path: Chemin vers l'image à traiter
        output_directory: Dossier où sauvegarder l'image traitée
        output_filename: Nom du fichier de sortie
        should_save: Si True, sauvegarde l'image traitée
        
    Returns:
        Image prétraitée (tableau numpy)
    """
    # Convertit l'image de BGR vers RGB
    processed_image_result = cv2.imread(image_file_path)[..., ::-1]
    
    # Applique le traitement deux fois pour améliorer la précision
    for iteration in range(2):
        resized_img, image_shape, scale_factor = yoco_resize_image_constant_area(processed_image_result)
        detected_lines = yoco_detect_chessboard_lines(resized_img)
        lattice_points_result = yoco_detect_lattice_points(resized_img, detected_lines)
        inner_corner_points = yoco_detect_inner_board_corners(resized_img, lattice_points_result, detected_lines)
        padded_corner_points = yoco_pad_board_corners(inner_corner_points, resized_img)
        
        try:
            processed_image_result = yoco_crop_chessboard_image(processed_image_result, padded_corner_points, scale_factor)
        except:
            print("ATTENTION: Impossible de recadrer autour des points extérieurs")
            processed_image_result = yoco_crop_chessboard_image(processed_image_result, inner_corner_points, scale_factor)
    
    if should_save:
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        plt.imsave("%s/%s" % (output_directory, output_filename), processed_image_result)
    
    return processed_image_result


def yoco_preprocess_single_chess_image(game_name, perspective_version, image_number):
    """
    Prétraite une seule image d'une partie d'échecs.
    
    Args:
        game_name: Nom de la partie
        perspective_version: Version de l'image ('white' ou 'black')
        image_number: Numéro de l'image dans la séquence
        
    Returns:
        True si le traitement a réussi, False sinon
    """
    raw_image_path_pattern = RAW_DATA_FOLDER + '%s/%s/%i.*' % (game_name, perspective_version, image_number)
    output_directory_path = PREPROCESSED_DATA_FOLDER + "%s/%s/" % (game_name, perspective_version)
    
    matching_image_files = glob.glob(raw_image_path_pattern)
    
    if not matching_image_files:
        print(f"ERREUR: Aucune image trouvée à {raw_image_path_pattern}")
        return False
    
    if len(matching_image_files) > 1:
        print(f"ATTENTION: Plusieurs fichiers trouvés, utilisation du premier: {matching_image_files[0]}")
    
    image_file_path = matching_image_files[0]
    print(f"Traitement: {image_file_path}")
    
    try:
        yoco_preprocess_chessboard_image(image_file_path, output_directory=output_directory_path,
                        output_filename="%i.png" % image_number, should_save=True)
        print(f"✓ Image sauvegardée avec succès dans {output_directory_path}{image_number}.png")
        return True
    except Exception as e:
        print(f"✗ ERREUR lors du traitement de l'image: {e}")
        import traceback
        traceback.print_exc()
        return False


def yoco_preprocess_multiple_chess_images(images_to_process_list):
    """
    Prétraite plusieurs images d'échiquiers.
    
    Args:
        images_to_process_list: Liste de tuples (game_name, version, image_number)
    """
    total_images_count = len(images_to_process_list)
    successful_count = 0
    failed_images_list = []
    
    print(f"Traitement de {total_images_count} images...")
    for game_name, version, image_number in images_to_process_list:
        current_progress_index = successful_count + len(failed_images_list) + 1
        print(f"\n[{current_progress_index}/{total_images_count}] {game_name}/{version}/image {image_number}")
        if yoco_preprocess_single_chess_image(game_name, version, image_number):
            successful_count += 1
        else:
            failed_images_list.append((game_name, version, image_number))
    
    print(f"\n{'='*50}")
    print(f"Résumé: {successful_count}/{total_images_count} images traitées avec succès")
    if failed_images_list:
        print(f"Images en échec ({len(failed_images_list)}):")
        for game, version, num in failed_images_list:
            print(f"  - {game}/{version}/image {num}")


def yoco_preprocess_chess_games_list(games_list):
    """
    Prétraite toutes les images d'une liste de parties d'échecs.
    
    Args:
        games_list: Liste des noms de parties à traiter
    """
    for game_name in games_list:
        for perspective_version in ['white', 'black']:
            image_files_list = []
            folder_path_pattern = RAW_DATA_FOLDER + '%s/%s/*' % (game_name, perspective_version)
            for file_path_item in glob.glob(folder_path_pattern):
                image_files_list.append(file_path_item)

            output_directory_path = PREPROCESSED_DATA_FOLDER + "%s/%s/" % (game_name, perspective_version)
            processed_image_count = 0
            
            # Trie les fichiers par numéro d'image
            image_files_list.sort(key=lambda file_path: int(file_path.split('/')[-1].split('.')[0]))
            
            for image_file_path_item in image_files_list:
                processed_image_count += 1
                yoco_preprocess_chessboard_image(image_file_path_item, output_directory=output_directory_path,
                                 output_filename="%i.png" % processed_image_count, should_save=True)
            
            if processed_image_count > 0:
                print("Terminé, images sauvegardées dans %s." % output_directory_path)
            else:
                print("Aucune image trouvée dans %s" % folder_path_pattern)


if __name__ == '__main__':
    # Parties déjà traitées: carlsen_anand_2014, carlsen_gukesh_2025, 
    # vachier-lagrave_carlsen_2023, david_vachier-lagrave_2014
    # game_list = ['david_vachier-lagrave_2014']
    # yoco_preprocess_chess_games_list(game_list)
    yoco_preprocess_single_chess_image('vachier-lagrave_carlsen_2023', 'black', 57)

