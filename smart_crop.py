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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_det_weights")
    parser.add_argument("aesthetics_weights")
    parser.add_argument("image")
    parser.add_argument("ratio")
    args = parser.parse_args()

    device = saillance.get_optimal_device()
    model = saillance.load_premade_model(args.feature_det_weights, device)
    r, initial_shape = saillance.inference(model, args.image, device)
    r = r * (255/np.max(r))
    r = (r > 50) * r
    
    plt.imshow(r, cmap='gray', vmin=0, vmax=255)
    plt.title("saliency result")
    plt.show()

    interest_clusters = points_interets.interest_clusters(r)

    potential_crops = []
    ratio = float(args.ratio)

    for i in range(len(interest_clusters)):
        crop_tuple = recadrage.get_crop_tuple_one_center(ratio, r, initial_shape, interest_clusters[i])
        potential_crops.append(recadrage.crop_image(args.image, crop_tuple))
    crop_tuple = recadrage.get_crop_tuple_using_least_square_distance_to_interest_points(ratio, r.shape, initial_shape, [i['centroid'] for i in interest_clusters])
    potential_crops.append(recadrage.crop_image(args.image, crop_tuple))

    crop_tuple = recadrage.get_crop_tuple_least_square_distance_to_best_interest_points(ratio, r.shape, initial_shape, interest_clusters)
    potential_crops.append(recadrage.crop_image(args.image, crop_tuple))

    crop_tuple = recadrage.get_crop_tuple_using_1D_saliency(ratio, r, initial_shape)
    potential_crops.append(recadrage.crop_image(args.image, crop_tuple))

    crop_tuple = recadrage.get_crop_tuple_random(ratio, r, initial_shape)
    potential_crops.append(recadrage.crop_image(args.image, crop_tuple))

    resize = transforms.Resize((256, 256), antialias=True)

    for i in range(len(potential_crops)):
        tmp = cv2.cvtColor(potential_crops[i], cv2.COLOR_BGR2RGB)
        tens = resize.forward(torch.mul(transforms.ToTensor()(tmp), 255.0).type(torch.IntTensor).unsqueeze(0))
        potential_crops[i] = tens
    
    inp = torch.Tensor(len(potential_crops), 3, 256, 256)
    torch.cat(potential_crops, out=inp)

    aes_model = aesthetics.load_premade_model(args.aesthetics_weights, device)
    res = aesthetics.inference_torch(aes_model, inp, device)

    cv2.waitKey(0)
    cv2.destroyAllWindows()