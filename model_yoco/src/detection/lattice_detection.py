import sys
import os
# Ajoute le répertoire racine du projet au path pour importer deps
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)
import deps

import numpy as np
import cv2
import collections
import scipy
import scipy.cluster
import tensorflow as tf

# Chemin vers le modèle LAPS (optionnel, utilise deps.laps si non disponible)
_laps_model_path = os.path.join(project_root, 'data', 'laps_models', 'laps.h5')
try:
    if os.path.exists(_laps_model_path):
        NEURAL_MODEL = tf.keras.models.load_model(_laps_model_path, compile=False)
        from tensorflow.keras.optimizers import RMSprop
        NEURAL_MODEL.compile(RMSprop(learning_rate=0.001),
                            loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'])
    else:
        raise FileNotFoundError(f"Model file not found: {_laps_model_path}")
except Exception as e:
    print(f"Warning: Could not load model from {_laps_model_path}: {e}")
    try:
        from deps.laps import model as NEURAL_MODEL
    except ImportError:
        print("Warning: Could not load model from deps.laps either. Some functionality may be limited.")
        NEURAL_MODEL = None


def laps_intersections(lines):
    __lines = [[(a[0], a[1]), (b[0], b[1])] for a, b in lines]
    return deps.geometry.isect_segments(__lines)


def laps_cluster(points, max_dist=10):
    Y = scipy.spatial.distance.pdist(points)
    Z = scipy.cluster.hierarchy.single(Y)
    T = scipy.cluster.hierarchy.fcluster(Z, max_dist, 'distance')
    clusters = collections.defaultdict(list)
    for i in range(len(T)):
        clusters[T[i]].append(points[i])
    clusters = clusters.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]),
                                np.mean(np.array(arr)[:, 1])), clusters)
    return list(clusters)


def laps_detector(img):
    global NC_LAYER

    hashid = str(hash(img.tostring()))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
    img = cv2.Canny(img, 0, 255)
    img = cv2.resize(img, (21, 21), interpolation=cv2.INTER_CUBIC)

    imgd = img

    X = [np.where(img > int(255/2), 1, 0).ravel()]
    X = X[0].reshape([-1, 21, 21, 1])

    img = cv2.dilate(img, None)
    mask = cv2.copyMakeBorder(img, top=1, bottom=1, left=1, right=1,
                              borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
    mask = cv2.bitwise_not(mask)
    i = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    _c = np.zeros((23, 23, 3), np.uint8)

    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        x, y = int(x), int(y)
        approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and radius < 14:
            cv2.drawContours(_c, [cnt], 0, (0, 255, 0), 1)
            i += 1
        else:
            cv2.drawContours(_c, [cnt], 0, (0, 0, 255), 1)

    if i == 4:
        return (True, 1)

    pred = NEURAL_MODEL.predict(X)
    a, b = pred[0][0], pred[0][1]
    t = a > b and b < 0.03 and a > 0.975

    if t:
        return (True, pred[0])
    else:
        return (False, pred[0])

################################################################################


def yoco_detect_lattice_points(image_array, line_segments, detection_size=10):
    """
    Détecte les points du réseau (lattice) de l'échiquier à partir des lignes détectées.
    
    Args:
        image_array: Image de l'échiquier
        line_segments: Liste des segments de lignes détectées
        detection_size: Taille de la fenêtre de détection
        
    Returns:
        Liste des points du réseau détectés
    """
    intersection_points, validated_points = laps_intersections(line_segments), []

    for point_coord in intersection_points:
        point_coord = list(map(int, point_coord))

        detection_x1 = max(0, int(point_coord[0]-detection_size-1))
        detection_x2 = max(0, int(point_coord[0]+detection_size))
        detection_y1 = max(0, int(point_coord[1]-detection_size))
        detection_y2 = max(0, int(point_coord[1]+detection_size+1))

        detection_image_patch = image_array[detection_y1:detection_y2, detection_x1:detection_x2]
        patch_shape = np.shape(detection_image_patch)

        if patch_shape[0] <= 0 or patch_shape[1] <= 0:
            continue

        detection_result = laps_detector(detection_image_patch)
        if not detection_result[0]:
            continue

        if point_coord[0] < 0 or point_coord[1] < 0:
            continue
        validated_points += [point_coord]
    clustered_points = laps_cluster(validated_points)

    return clustered_points
