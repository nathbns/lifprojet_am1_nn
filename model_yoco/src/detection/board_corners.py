import sys
import os
# Ajoute le répertoire src au path pour les imports relatifs
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from detection.lattice_detection import laps_intersections, laps_cluster
from detection.line_detection import slid_tendency
import scipy
import cv2
import pyclipper
import numpy as np
import matplotlib.path
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import collections
import itertools
import random
import math
import sklearn.cluster
from copy import copy
na = np.array


def llr_normalize(points): return [[int(a), int(b)] for a, b in points]


def llr_correctness(points, shape):
    __points = []
    for pt in points:
        if pt[0] < 0 or pt[1] < 0 or \
            pt[0] > shape[1] or \
                pt[1] > shape[0]:
            continue
        __points += [pt]
    return __points


def llr_unique(a):
    indices = sorted(range(len(a)), key=a.__getitem__)
    indices = set(next(it) for k, it in
                  itertools.groupby(indices, key=a.__getitem__))
    return [x for i, x in enumerate(a) if i in indices]


def llr_polysort(pts):
    mlat = sum(x[0] for x in pts) / len(pts)
    mlng = sum(x[1] for x in pts) / len(pts)

    def __sort(x):
        return (math.atan2(x[0]-mlat, x[1]-mlng) +
                2*math.pi) % (2*math.pi)
    pts.sort(key=__sort)
    return pts


def llr_polyscore(cnt, pts, cen, alfa=5, beta=2):
    a = cnt[0]
    b = cnt[1]
    c = cnt[2]
    d = cnt[3]

    area = cv2.contourArea(cnt)
    t2 = area < (4 * alfa * alfa) * 5
    if t2:
        return 0

    gamma = alfa/1.5

    pco = pyclipper.PyclipperOffset()
    pco.AddPath(cnt, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    pcnt = matplotlib.path.Path(pco.Execute(gamma)[0])
    wtfs = pcnt.contains_points(pts)
    pts_in = min(np.count_nonzero(wtfs), 49)
    t1 = pts_in < min(len(pts), 49) - 2 * beta - 1
    if t1:
        return 0

    A = pts_in
    B = area

    def nln(l1, x, dx): return \
        np.linalg.norm(np.cross(na(l1[1])-na(l1[0]),
                                na(l1[0])-na(x)))/dx
    pcnt_in = []
    i = 0
    for pt in wtfs:
        if pt:
            pcnt_in += [pts[i]]
        i += 1

    def __convex_approx(points, alfa=0.001):
        hull = scipy.spatial.ConvexHull(na(points)).vertices
        cnt = na([points[pt] for pt in hull])
        return cnt

    cnt_in = __convex_approx(na(pcnt_in))

    points = cnt_in
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    cen2 = (sum(x) / len(points),
            sum(y) / len(points))

    G = np.linalg.norm(na(cen)-na(cen2))

    a = [cnt[0], cnt[1]]
    b = [cnt[1], cnt[2]]
    c = [cnt[2], cnt[3]]
    d = [cnt[3], cnt[0]]
    lns = [a, b, c, d]
    E = 0
    F = 0
    for l in lns:
        d = np.linalg.norm(na(l[0])-na(l[1]))
        for p in cnt_in:
            r = nln(l, p, d)
            if r < gamma:
                E += r
                F += 1
    if F == 0:
        return 0
    E /= F

    if B == 0 or A == 0:
        return 0

    C = 1+(E/A)**(1/3)
    D = 1+(G/A)**(1/5)
    R = (A**4)/((B**2) * C * D)

    return R


def yoco_detect_inner_board_corners(image_array, lattice_points, line_segments):
    """
    Détecte les coins intérieurs de l'échiquier à partir des points du réseau et des lignes.
    
    Args:
        image_array: Image de l'échiquier
        lattice_points: Points du réseau détectés
        line_segments: Segments de lignes détectés
        
    Returns:
        Liste des quatre coins intérieurs de l'échiquier
    """
    original_points = lattice_points

    def __convex_approx(points, alfa=0.01):
        hull = scipy.spatial.ConvexHull(na(points)).vertices
        cnt = na([points[pt] for pt in hull])
        approx = cv2.approxPolyDP(cnt, alfa *
                                  cv2.arcLength(cnt, True), True)
        return llr_normalize(itertools.chain(*approx))

    __cache = {}

    def __dis(a, b):
        idx = hash("__dis" + str(a) + str(b))
        if idx in __cache:
            return __cache[idx]
        __cache[idx] = np.linalg.norm(na(a)-na(b))
        return __cache[idx]

    def nln(l1, x, dx): return \
        np.linalg.norm(np.cross(na(l1[1])-na(l1[0]),
                                na(l1[0])-na(x)))/dx

    pregroup = [[], []]
    score_dict = {}

    lattice_points = llr_correctness(llr_normalize(lattice_points), image_array.shape)

    clustered_points_dict = {}
    lattice_points = llr_polysort(lattice_points)
    max_cluster_size, largest_cluster_points = 0, []
    alpha_parameter = math.sqrt(cv2.contourArea(na(lattice_points))/49)
    clustering_result = sklearn.cluster.DBSCAN(eps=alpha_parameter*4).fit(lattice_points)
    for point_index in range(len(lattice_points)):
        clustered_points_dict[point_index] = []
    for point_index in range(len(lattice_points)):
        if clustering_result.labels_[point_index] != -1:
            clustered_points_dict[clustering_result.labels_[point_index]] += [lattice_points[point_index]]
    for point_index in range(len(lattice_points)):
        if len(clustered_points_dict[point_index]) > max_cluster_size:
            max_cluster_size = len(clustered_points_dict[point_index])
            largest_cluster_points = clustered_points_dict[point_index]
    if len(clustered_points_dict) > 0 and len(lattice_points) > 49/2:
        lattice_points = largest_cluster_points

    convex_ring = __convex_approx(llr_polysort(lattice_points))

    point_count = len(lattice_points)
    beta_parameter = point_count*(5/100)
    alpha_parameter = math.sqrt(cv2.contourArea(na(lattice_points))/49)

    x_coords = [point[0] for point in lattice_points]
    y_coords = [point[1] for point in lattice_points]
    centroid_point = (sum(x_coords) / len(lattice_points),
                sum(y_coords) / len(lattice_points))

    def __v(l):
        y_0, x_0 = l[0][0], l[0][1]
        y_1, x_1 = l[1][0], l[1][1]

        x_2 = 0
        t = (x_0-x_2)/(x_0-x_1+0.0001)
        a = [int((1-t)*x_0+t*x_1), int((1-t)*y_0+t*y_1)][::-1]

        x_2 = image_array.shape[0]
        t = (x_0-x_2)/(x_0-x_1+0.0001)
        b = [int((1-t)*x_0+t*x_1), int((1-t)*y_0+t*y_1)][::-1]

        poly1 = llr_polysort([[0, 0], [0, image_array.shape[0]], a, b])
        s1 = llr_polyscore(na(poly1), lattice_points, centroid_point, beta=beta_parameter, alfa=alpha_parameter/2)
        poly2 = llr_polysort([a, b,
                              [image_array.shape[1], 0], [image_array.shape[1], image_array.shape[0]]])
        s2 = llr_polyscore(na(poly2), lattice_points, centroid_point, beta=beta_parameter, alfa=alpha_parameter/2)

        return [a, b], s1, s2

    def __h(l):
        x_0, y_0 = l[0][0], l[0][1]
        x_1, y_1 = l[1][0], l[1][1]

        x_2 = 0
        t = (x_0-x_2)/(x_0-x_1+0.0001)
        a = [int((1-t)*x_0+t*x_1), int((1-t)*y_0+t*y_1)]

        x_2 = image_array.shape[1]
        t = (x_0-x_2)/(x_0-x_1+0.0001)
        b = [int((1-t)*x_0+t*x_1), int((1-t)*y_0+t*y_1)]

        poly1 = llr_polysort([[0, 0], [image_array.shape[1], 0], a, b])
        s1 = llr_polyscore(na(poly1), lattice_points, centroid_point, beta=beta_parameter, alfa=alpha_parameter/2)
        poly2 = llr_polysort([a, b,
                              [0, image_array.shape[0]], [image_array.shape[1], image_array.shape[0]]])
        s2 = llr_polyscore(na(poly2), lattice_points, centroid_point, beta=beta_parameter, alfa=alpha_parameter/2)

        return [a, b], s1, s2

    for line_seg in line_segments:
        for point_coord in lattice_points:
            t1 = nln(line_seg, point_coord, __dis(*line_seg)) < alpha_parameter
            t2 = nln(line_seg, centroid_point, __dis(*line_seg)) > alpha_parameter * 2.5

            if t1 and t2:
                delta_x, delta_y = line_seg[0][0]-line_seg[1][0], line_seg[0][1]-line_seg[1][1]
                if abs(delta_x) < abs(delta_y):
                    extended_line, s1, s2 = __v(line_seg)
                    orientation = 0
                else:
                    extended_line, s1, s2 = __h(line_seg)
                    orientation = 1
                if s1 == 0 and s2 == 0:
                    continue
                pregroup[orientation] += [extended_line]

    pregroup[0] = llr_unique(pregroup[0])
    pregroup[1] = llr_unique(pregroup[1])

    for vertical_pair in itertools.combinations(pregroup[0], 2):
        for horizontal_pair in itertools.combinations(pregroup[1], 2):
            polygon_corners = laps_intersections([vertical_pair[0], vertical_pair[1], horizontal_pair[0], horizontal_pair[1]])
            polygon_corners = llr_correctness(polygon_corners, image_array.shape)
            if len(polygon_corners) != 4:
                continue
            polygon_corners = na(llr_polysort(llr_normalize(polygon_corners)))
            if not cv2.isContourConvex(polygon_corners):
                continue
            score_dict[-llr_polyscore(polygon_corners, lattice_points, centroid_point,
                             beta=beta_parameter, alfa=alpha_parameter/2)] = polygon_corners

    score_dict = collections.OrderedDict(sorted(score_dict.items()))
    best_score_key = next(iter(score_dict))
    four_corner_points = llr_normalize(score_dict[best_score_key])

    return four_corner_points


def yoco_pad_board_corners(corner_points, image_array):
    """
    Ajoute du padding autour des coins de l'échiquier.
    
    Args:
        corner_points: Quatre points définissant les coins
        image_array: Image de l'échiquier
        
    Returns:
        Points avec padding appliqué
    """
    padding_clipper = pyclipper.PyclipperOffset()
    padding_clipper.AddPath(corner_points, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)

    padded_points = padding_clipper.Execute(60)[0]

    return padded_points
