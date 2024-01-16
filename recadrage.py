import cv2
import numpy as np
from scipy.optimize import minimize
import random

def crop_image(image, crop_tuple):
    img = cv2.imread(image)
    x, y, w, h = crop_tuple
    return img[x:x+w, y:y+h]

def get_crop_tuple_using_1D_saliency(ratio, saliency, initial_shape):
    rows, cols = saliency.shape[0], saliency.shape[1]
    if cols == int(rows*ratio):
        return (0, 0, initial_shape[1], initial_shape[0])
    # 1D sliding windows on rows
    if cols < int(rows*ratio):
        sumrows = np.sum(saliency, 1)
        nrows = int(cols/ratio)
        max_s = np.sum(sumrows[0:nrows])
        tmp_s = max_s
        s = [tmp_s]
        for r in range(1, rows-nrows+1):
            tmp_s = tmp_s - sumrows[r-1] + sumrows[r+nrows-1]
            s.append(tmp_s)
            if tmp_s > max_s:
                max_s = tmp_s
        ind = np.where(np.array(s) == max_s)[0]
        best_r = ind[len(ind)//2]
        return (int(best_r/rows*initial_shape[1]), 0, int(initial_shape[0]/ratio), initial_shape[0])
    
    # 1D sliding windows on cols
    sumcols = np.sum(saliency, 0)
    ncols = int(rows * ratio)
    s = []
    max_s = np.sum(sumcols[0:ncols])
    tmp_s = max_s
    for c in range(1, cols-ncols+1):
        tmp_s = tmp_s - sumcols[c-1] + sumcols[c+ncols-1]
        s.append(tmp_s)
        if tmp_s > max_s:
            max_s = tmp_s
    ind = np.where(np.array(s) == max_s)[0]
    best_c = ind[len(ind)//2]
    return (0, int(best_c/cols*initial_shape[0]), initial_shape[1], int(initial_shape[1]*ratio))

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
        Y2 = Y1 + (X2-X1)*ratio
        Y2 = max(min(Y2, args[0]), 0)
        
        for c in centroids:
            XC = c[0]
            YC = c[1]
            if XC < X1 or XC > 2:
                distance += ((X2-X1)*XC**2-XC*(X2**2-X1**2)+1/3*(X2**3-X1**3)) / (X2 - X1)**2
            else:
                distance += ((XC-X1)*XC**2-XC*(XC**2-X1**2)+1/3*(XC**3-X1**3)) / (X2 - X1)**2
                distance += ((X2-XC)*XC**2-XC*(X2**2-XC**2)+1/3*(X2**3-XC**3)) / (X2 - X1)**2
            if YC < Y1 or YC > 2:
                distance += ((Y2-Y1)*YC**2-YC*(Y2**2-Y1**2)+1/3*(Y2**3-Y1**3)) / (Y2 - Y1)**2
            else:
                distance += ((YC-Y1)*YC**2-YC*(YC**2-Y1**2)+1/3*(YC**3-Y1**3)) / (Y2 - Y1)**2
                distance += ((Y2-YC)*YC**2-YC*(Y2**2-YC**2)+1/3*(Y2**3-YC**3)) / (Y2 - Y1)**2
        return np.sqrt(distance)

def get_crop_tuple_using_least_square_distance_to_interest_points(ratio, saliency_shape, initial_shape, centroids):
    c_scaled = []
    for c in centroids:
        c_scaled.append([c[0]*initial_shape[1]/saliency_shape[0], c[1]*initial_shape[0]/saliency_shape[1]])
    initial_params = [0, initial_shape[1], 0]  # Initial scale is 1
    opt_result = minimize(square_distance_sum, initial_params, (initial_shape[0], initial_shape[1], c_scaled, ratio), method='Nelder-Mead')

    # Extract optimized parameters
    p = opt_result.x

    X1, X2, Y1 = p[0], p[1], p[2]
    X1 = min(max(X1, 0), initial_shape[1])
    X2 = max(min(X2, initial_shape[1]), 0)
    Y1 = max(min(Y1, initial_shape[0]), 0)
    if X1 > X2:
        X2, X1 = X1, X2

    x, y, w, h = int(X1), int(Y1), int(X2-X1), int((X2-X1)*ratio)

    return (x, y, w, h)

def get_crop_tuple_least_square_distance_to_best_interest_points(ratio, saliency_shape, initial_shape, centroids_data):
    centroids_avg_saliency = np.array(map(lambda centroid: centroid['avg_saliency'], centroids_data))
    best_interest_indexes = np.argpartition(centroids_avg_saliency, -3)[-3:]
    best_centroids = [centroids_data[idx]['centroid'] for idx in best_interest_indexes]
    return get_crop_tuple_using_least_square_distance_to_interest_points(ratio, saliency_shape, initial_shape, best_centroids)

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
    print(centroid_map['centroid'])
    print(saliency_map.shape)
    print(initial_shape)
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
    
    return (x,y,w,h)

def get_random_point_based_on_saliency(saliency_map):
    rows, columns = np.where(saliency_map > 0)
    points_array = np.asarray(list(zip(rows, columns)))
    total_saliency = np.sum(saliency_map)
    probabilities = [saliency_map[col][row] / total_saliency for row, col in points_array]
    random_index = random.choices(points_array, probabilities)[0]
    return points_array[random_index]

# CASINO
def get_crop_tuple_random(ratio, saliency_map, initial_shape):
    random_point = get_random_point_based_on_saliency(saliency_map) # row, column
    max_height = min(saliency_map.shape[0], int(saliency_map.shape[1] / ratio))
    random_crop_height = int(random.uniform(0.5 * max_height, max_height))
    random_crop_width = int(random_crop_height * ratio)
    
    crop_start_X, crop_start_Y = int(random_point[1] - random_crop_width / 2), int(random_point[0] - random_crop_height / 2)
    
    if random_point[1] + 1 > saliency_map.shape[1] - random_crop_width / 2:
        crop_start_X = int(saliency_map.shape[1] - random_crop_width / 2 - 1)
    elif random_point[1] + 1 > random_crop_width / 2:
        crop_start_X = 0

    if random_point[0] + 1 > saliency_map.shape[0] - random_crop_height / 2:
        crop_start_Y = int(saliency_map.shape[0] - random_crop_height / 2 - 1)
    elif random_point[0] + 1 > random_crop_height / 2:
        crop_start_Y = 0

    return crop_start_X, random_crop_width, crop_start_Y, random_crop_height

if __name__ == "__main__":
    # Example usage:
    ratio = 0.1
    crop_tuple = get_crop_tuple_using_least_square_distance_to_interest_points(1.7, (540, 960), (960, 540), [[344.12998125974815, 656.0959556507776], [157.68953265450583, 310.13154991381725]])
    print(crop_tuple)
    cv2.imshow("Image Recadrée", crop_image(r"..\..\Downloads\image-attractive-960x540.jpg", crop_tuple))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
