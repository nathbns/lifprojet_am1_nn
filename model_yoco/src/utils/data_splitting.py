"""
Utilitaires pour la division des données en ensembles d'entraînement, validation et test.
"""
from pathlib import Path
from shutil import copy
import os
import random
import sys
import os
# Ajoute le répertoire src au path pour les imports relatifs
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, src_dir)

from constant import (
    LABELED_DATA_FOLDER, CNN_DATA_FOLDER,
    TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT
)


def yoco_split_dataset_into_sets():
    """
    Divise les données labellisées en ensembles d'entraînement, validation et test.
    Les images sont réparties aléatoirement selon les proportions définies dans les constantes.
    """
    # Vérifie que les proportions sont valides (somme = 1)
    assert abs((TRAIN_SPLIT + VALIDATION_SPLIT + TEST_SPLIT) - 1) < 1e-8
    
    total_processed_count = 0
    for directory_path, subdirs, file_list in os.walk(LABELED_DATA_FOLDER):
        for filename in [f for f in file_list if f.endswith(".jpg")]:
            random_split_value = random.uniform(0, 1)
            piece_label = directory_path.split('/')[-1]
            
            # Détermine dans quel ensemble placer l'image
            if random_split_value < TRAIN_SPLIT:
                dataset_split_type = 'train'
            elif random_split_value < TRAIN_SPLIT + VALIDATION_SPLIT:
                dataset_split_type = 'validation'
            else:
                dataset_split_type = 'test'
            
            # Crée le dossier de destination et copie l'image
            destination_directory = CNN_DATA_FOLDER + '%s/%s/' % (dataset_split_type, piece_label)
            Path(destination_directory).mkdir(parents=True, exist_ok=True)
            copy(directory_path + '/' + filename, destination_directory)
            total_processed_count += 1
    
    print(f"Répartition terminée: {total_processed_count} images copiées")


if __name__ == '__main__':
    yoco_split_dataset_into_sets()

