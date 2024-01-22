# Projet IA

## Content of the repository

This repository contains 5 main files for the application:
- `saillance.py`: contains all the code necessary to run the saliency detection on an image.
- `points_interets.py`: contains all the code necessary to extract interest points from saliency map.
- `recadrage.py`: contains all the code used to crop an image based on a provided ratio and saliency informations (saliency map or interest points).
- `aesthetics.py`: contains all the code necessary to run the aesthetics scoring on an image.
- `smart_crop.py`: the main file of the application that combines all the steps and ensures the production of the final result using user input 

## Install dependencies

Start by installing the require python packages using:
```
python3 -m pip install -r requirements.txt
```

This can be executed in a venv in order to limit these packages to a venv.

## Retrieve weights

In order for the algorithm to work, one must first retrieve the weights of the saliency detection neural network and the aesthetics scoring neural network.
The weights are available [here](https://github.com/LeoMarche/ProjetIA/releases/download/poc/premade.pth) and here (TODO upload and insert link).

## Testing the Cropping Modules
3 cropping files are proposed : 2 are tested for testing Functional Cropping the third is in the part of " How to use the final product"
-python3 smart_crop2.py premade.pth test.jpg 1.5
-python3 smart_crop3.py premade.pth test.jpg 1.5

## Comparing Algorithms time of execution:
it's not fixed , it's changes 



## How to use the final product

Once the dependencies installed, run `python3 smart_crop.py <path_to_saliency_weights> <path_to_aesthetics_weights> <path_to_image> <resize_ratio>`  
It will make windows appear to show intermediate results, close them to continue the algorithm

Example: `python3 smart_crop.py model1.pth model2.pth imgtest1.jpg 1.5` in order to run the algorithm and to intelligently crop the imgtest1.jpg image at a 1.5 ratio.
