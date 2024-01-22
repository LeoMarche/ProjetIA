import cv2
import numpy as np
from scipy.optimize import minimize
import random


def crop_image(image, crop_tuple):
    img = cv2.imread(image)
    x, y, w, h = crop_tuple
    return img[x:x + w, y:y + h]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_crop_tuple_sigmoid_based(ratio, saliency_map, initial_shape):
    rows, cols = saliency_map.shape

    # Appliquer la fonction logistique aux indices des lignes et colonnes
    sigmoid_rows = sigmoid(np.arange(rows))
    sigmoid_cols = sigmoid(np.arange(cols))

    # Calculer les nouvelles dimensions du crop
    crop_width = int(cols * 0.8)  # Exemple : 80% de la largeur initiale
    crop_height = int(crop_width / ratio)

    # Appliquer la fonction logistique aux dimensions du crop
    sigmoid_crop_width = sigmoid(np.arange(crop_width))
    sigmoid_crop_height = sigmoid(np.arange(crop_height))

    # Calculer les nouvelles coordonnées du coin supérieur gauche
    x = int((rows - crop_height) * np.mean(sigmoid_rows))
    y = int((cols - crop_width) * np.mean(sigmoid_cols))

    return x, y, crop_height, crop_width

def square_distance_sum(params, *args):
    centroids = args[2]
    ratio = args[3]
    distance = 0
    X1, X2, Y1 = params
    X1 = min(max(X1, 0), args[1])
    X2 = max(min(X2, args[1]), 0)
    Y1 = max(min(Y1, args[0]), 0)
    if X1 > X2:
        X2, X1 = X1, X2
    Y2 = Y1 + (X2 - X1) * ratio
    Y2 = max(min(Y2, args[0]), 0)

    for c in centroids:
        XC = c[0]
        YC = c[1]
        if XC < X1 or XC > 2:
            distance += ((X2 - X1) * XC ** 2 - XC * (X2 ** 2 - X1 ** 2) + 1 / 3 * (X2 ** 3 - X1 ** 3)) / (X2 - X1) ** 2
        else:
            distance += ((XC - X1) * XC ** 2 - XC * (XC ** 2 - X1 ** 2) + 1 / 3 * (XC ** 3 - X1 ** 3)) / (X2 - X1) ** 2
            distance += ((X2 - XC) * XC ** 2 - XC * (X2 ** 2 - XC ** 2) + 1 / 3 * (X2 ** 3 - XC ** 3)) / (X2 - X1) ** 2
        if YC < Y1 or YC > 2:
            distance += ((Y2 - Y1) * YC ** 2 - YC * (Y2 ** 2 - Y1 ** 2) + 1 / 3 * (Y2 ** 3 - Y1 ** 3)) / (Y2 - Y1) ** 2
        else:
            distance += ((YC - Y1) * YC ** 2 - YC * (YC ** 2 - Y1 ** 2) + 1 / 3 * (YC ** 3 - Y1 ** 3)) / (Y2 - Y1) ** 2
            distance += ((Y2 - YC) * YC ** 2 - YC * (Y2 ** 2 - YC ** 2) + 1 / 3 * (Y2 ** 3 - YC ** 3)) / (Y2 - Y1) ** 2
    return np.sqrt(distance)


def get_crop_tuple_least_square_distance_to_interest_points(ratio, saliency_shape, initial_shape, centroids):
    c_scaled = []
    for c in centroids:
        c_scaled.append([c[0] * initial_shape[1] / saliency_shape[0], c[1] * initial_shape[0] / saliency_shape[1]])
    initial_params = [0, initial_shape[1], 0]  # Initial scale is 1
    opt_result = minimize(square_distance_sum, initial_params, (initial_shape[0], initial_shape[1], c_scaled, ratio),
                          method='Nelder-Mead')

    # Extract optimized parameters
    p = opt_result.x

    X1, X2, Y1 = p[0], p[1], p[2]
    X1 = min(max(X1, 0), initial_shape[1])
    X2 = max(min(X2, initial_shape[1]), 0)
    Y1 = max(min(Y1, initial_shape[0]), 0)
    if X1 > X2:
        X2, X1 = X1, X2

    x, y, w, h = int(X1), int(Y1), int(X2 - X1), int((X2 - X1) * ratio)

    return (x, y, w, h)


def get_crop_tuple_least_square_distance_to_best_interest_points(ratio, saliency_shape, initial_shape, centroids_data,
                                                                 n_best_interest_points):
    centroids_avg_saliency = np.array(list(map(lambda centroid: centroid['avg_saliency'], centroids_data)))
    best_interest_indexes = np.argpartition(centroids_avg_saliency, -n_best_interest_points)[-n_best_interest_points:]
    best_centroids = [centroids_data[idx]['centroid'] for idx in best_interest_indexes]
    return get_crop_tuple_least_square_distance_to_interest_points(ratio, saliency_shape, initial_shape, best_centroids)


def compute_average_weighted_distance(points, weights, centroid):
    dist = 0
    for ipo in range(len(points)):
        po = points[ipo]
        dist += np.linalg.norm(np.array(po) - np.array(centroid)) ** 2 * weights[ipo]
    return np.sqrt(dist / np.sum(weights))


def get_crop_tuple_one_center(ratio, saliency_map, initial_shape, centroid_map):
    pixel_coords = centroid_map['pixel_coords']
    weights = [saliency_map[c[0]][c[1]] for c in pixel_coords]
    av_dist = compute_average_weighted_distance(pixel_coords, weights, centroid_map['centroid'])

    av_dist_ratio = av_dist * 5

    if ratio > 1:
        ideal_row_nr = av_dist_ratio
        ideal_col_nr = ideal_row_nr * ratio
    else:
        ideal_col_nr = av_dist_ratio
        ideal_row_nr = ideal_col_nr / ratio

    if centroid_map['centroid'][0] + ideal_row_nr / 2 > saliency_map.shape[0]:
        rat = saliency_map.shape[0] - centroid_map['centroid'][0] * 2 / ideal_row_nr
        ideal_row_nr *= rat
        ideal_col_nr *= rat
    if centroid_map['centroid'][0] - ideal_row_nr / 2 < 0:
        rat = centroid_map['centroid'][0] * 2 / ideal_row_nr
        ideal_row_nr *= rat
        ideal_col_nr *= rat
    if centroid_map['centroid'][1] + ideal_col_nr / 2 > saliency_map.shape[1]:
        rat = saliency_map.shape[1] - centroid_map['centroid'][1] * 2 / ideal_col_nr
        ideal_row_nr *= rat
        ideal_col_nr *= rat
    if centroid_map['centroid'][1] - ideal_col_nr / 2 < 0:
        rat = centroid_map['centroid'][1] * 2 / ideal_col_nr
        ideal_row_nr *= rat
        ideal_col_nr *= rat

    x = int((centroid_map['centroid'][0] - ideal_row_nr / 2) * initial_shape[1] / saliency_map.shape[0])
    y = int((centroid_map['centroid'][1] - ideal_col_nr / 2) * initial_shape[0] / saliency_map.shape[1])
    w = int(ideal_row_nr * initial_shape[1] / saliency_map.shape[0])
    h = int(ideal_col_nr * initial_shape[0] / saliency_map.shape[1])

    return (x, y, w, h)


def get_random_point_based_on_saliency(saliency_map):
    rows, columns = np.where(saliency_map > 0)
    points_array = np.asarray(list(zip(rows, columns)))
    total_saliency = np.sum(saliency_map)
    probabilities = [saliency_map[row][col] / total_saliency for row, col in points_array]
    return random.choices(points_array, probabilities)[0]


def get_crop_tuple_random(ratio, saliency_map, initial_shape):
    random_point = get_random_point_based_on_saliency(saliency_map)  # row, column
    max_width = min(saliency_map.shape[0], int(saliency_map.shape[1] / ratio))
    crop_width = int(random.uniform(0.5 * max_width, max_width))
    crop_height = int(crop_width * ratio)

    crop_start_X, crop_start_Y = int(random_point[0] - crop_width / 2), int(random_point[1] - crop_height / 2)

    if random_point[0] + 1 > saliency_map.shape[0] - crop_width / 2:
        crop_start_X = int(saliency_map.shape[0] - crop_width - 1)
    elif random_point[0] + 1 < crop_width / 2:
        crop_start_X = 0

    if random_point[1] + 1 > saliency_map.shape[1] - crop_height / 2:
        crop_start_Y = int(saliency_map.shape[1] - crop_height - 1)
    elif random_point[1] + 1 < crop_height / 2:
        crop_start_Y = 0

    rescale_ratio_X = initial_shape[1] / saliency_map.shape[0]
    rescale_ratio_Y = initial_shape[0] / saliency_map.shape[1]

    crop_start_X = int(crop_start_X * rescale_ratio_X)
    crop_start_Y = int(crop_start_Y * rescale_ratio_Y)
    crop_width = int(crop_width * rescale_ratio_X)
    crop_height = int(crop_height * rescale_ratio_Y)

    return crop_start_X, crop_start_Y, crop_width, crop_height
