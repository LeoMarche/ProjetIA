import cv2
import numpy as np
from scipy.optimize import minimize

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
        X1 = min(max(X1, 0), args[0])
        X2 = max(min(X2, args[0]), 0)
        Y1 = max(min(Y1, args[1]), 0)
        if X1 > X2:
            X2, X1 = X1, X2
        Y2 = Y1 + (X2-X1)*ratio
        
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
    X1 = min(max(X1, 0), initial_shape[0])
    X2 = max(min(X2, initial_shape[0]), 0)
    Y1 = max(min(Y1, initial_shape[1]), 0)
    if X1 > X2:
        X2, X1 = X1, X2

    x, y, w, h = int(X1), int(Y1), int(X2-X1), int((X2-X1)*ratio)

    return (x, y, w, h)

if __name__ == "__main__":
    # Example usage:
    ratio = 0.1
    crop_tuple = get_crop_tuple_using_least_square_distance_to_interest_points(1.7, (540, 960), (960, 540), [[344.12998125974815, 656.0959556507776], [157.68953265450583, 310.13154991381725]])
    print(crop_tuple)
    cv2.imshow("Image Recadrée", crop_image(r"..\..\Downloads\image-attractive-960x540.jpg", crop_tuple))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
