import cv2
import numpy as np
from scipy.optimize import minimize

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

# Example usage:
ratio = 0.1
image = cv2.imread(r"imgest2.jpg")
centroids = np.array([(0,0), (2500,1667)])
Idef_result = apply_recadrage(image, centroids, ratio)
