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
    

class ImageTransformation:
    def apply_transform(self, src, params):
        # Extract parameters
        Tx, Ty, Theta, scale = params

        # Get image dimensions
        rows, cols = src.shape[:2]

        # Define the transformation matrix for translation, rotation, and scaling
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), Theta, scale)
        M[0, 2] += Tx
        M[1, 2] += Ty

        # Apply the transformation to the image
        dst = cv2.warpAffine(src, M, (cols, rows))

        return dst

def apply_recadrage(image, centroids, ratio):
    # Define the transformation object before using it in the optimization function
    transformation = ImageTransformation()

    def optimization_objective(params, *args):
        # Extract parameters and call the scoring function
        Tx, Ty, _, scale = params
        cropped_image = transformation.apply_transform(image, params)
        rows, cols = image.shape[:2]
        print(rows, cols)
        distances = np.sqrt((centroids[:, 0] - Tx) ** 2 + (centroids[:, 1] - Ty) ** 2)
        return np.sum(distances)
    
    def optimization_objective2(params, *args):
        distance = 0
        X1, X2, Y1 = params
        X1 = min(max(X1, 0), 2500)
        X2 = max(min(X2, 2500), 0)
        Y1 = max(min(Y1, 1667), 0)
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
            
        print(distance)
        return np.sqrt(distance)

    # Flatten the initial parameters
    initial_params = [0, 2500, 0]  # Initial scale is 1

    opt_result = minimize(optimization_objective2, initial_params, method='Nelder-Mead')

    # Extract optimized parameters
    p = opt_result.x
    print(p)

    X1, X2, Y1 = p[0], p[1], p[2]
    X1 = min(max(X1, 0), 2500)
    X2 = max(min(X2, 2500), 0)
    Y1 = max(min(Y1, 1667), 0)
    if X1 > X2:
        X2, X1 = X1, X2

    x, y, w, h = int(X1), int(Y1), int(X2-X1), int((X2-X1)*ratio)
    roi = image[y:y+h, x:x+w]
    print(x,y,h,w)
    cv2.imshow("Image Recadrée", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return roi

if __name__ == "__main__":
    # Example usage:
    ratio = 0.1
    image = cv2.imread(r"imgest2.jpg")
    centroids = np.array([(0,0), (2500,1667)])
    Idef_result = apply_recadrage(image, centroids, ratio)
