# intelligent image cropping application
import argparse
import matplotlib.pyplot as plt
import saillance
import points_interets
import recadrage
import aesthetics
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch
import os
import datetime

## Returns either the cpu if nod GPU is available or the first CUDA capable device else
def get_optimal_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_crops(crops, aesthetics_scores):
    if not(os.path.isdir("./output")):
        os.mkdir("./output")
    time = datetime.datetime.now()
    datetime_string = time.strftime("%d_%m_%Y_%H_%M_%S")
    for crop_idx in range(len(crops)):
        filename = "./output/output-" + datetime_string 
        filename += "-choix" + str(crop_idx + 1)
        filename += "-" + str(aesthetics_scores[crop_idx]) + ".jpg"
        cv2.imwrite(filename, crops[crop_idx])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_det_weights")
    parser.add_argument("aesthetics_weights")
    parser.add_argument("image")
    parser.add_argument("ratio")
    args = parser.parse_args()

    device = get_optimal_device()
    model = saillance.load_premade_model(args.feature_det_weights, device)
    saliency_map, initial_shape = saillance.inference(model, args.image, device)
    saliency_map = saliency_map * (255/np.max(saliency_map))
    saliency_map = (saliency_map > 50) * saliency_map
    
    plt.imshow(saliency_map, cmap='gray', vmin=0, vmax=255)
    plt.title("saliency result")
    plt.show()

    interest_clusters = points_interets.interest_clusters(saliency_map)

    potential_crops = []
    ratio = float(args.ratio)

    def add_potential_crop(crop_tuple):
        potential_crops.append(recadrage.crop_image(args.image, crop_tuple))

    for interest_cluster in interest_clusters:
        add_potential_crop(recadrage.get_crop_tuple_one_center(ratio, saliency_map, initial_shape, interest_cluster))

    centroids = [i['centroid'] for i in interest_clusters]
    if len(centroids) > 1: 
        add_potential_crop(recadrage.get_crop_tuple_least_square_distance_to_interest_points(ratio, saliency_map.shape, initial_shape, centroids))
    
    n_best_interest_points = 2
    if n_best_interest_points < len(centroids):
        add_potential_crop(recadrage.get_crop_tuple_least_square_distance_to_best_interest_points(ratio, saliency_map.shape, initial_shape, interest_clusters, n_best_interest_points))
    
    add_potential_crop(recadrage.get_crop_tuple_using_1D_saliency(ratio, saliency_map, initial_shape))
    
    add_potential_crop(recadrage.get_crop_tuple_random(ratio, saliency_map, initial_shape))

    crop_images = potential_crops[:]

    resize = transforms.Resize((256, 256), antialias=True)

    for interest_cluster in range(len(potential_crops)):
        tmp = cv2.cvtColor(potential_crops[interest_cluster], cv2.COLOR_BGR2RGB)
        tens = resize.forward(torch.mul(transforms.ToTensor()(tmp), 255.0).type(torch.IntTensor).unsqueeze(0))
        potential_crops[interest_cluster] = tens
    
    inp = torch.Tensor(len(potential_crops), 3, 256, 256)
    torch.cat(potential_crops, out=inp)

    aes_model = aesthetics.load_premade_model(args.aesthetics_weights, device)
    aesthetics_scores = aesthetics.inference_torch(aes_model, inp, device)

    save_crops(crop_images, aesthetics_scores)

    cv2.waitKey(0)
    cv2.destroyAllWindows()