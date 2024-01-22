# Projet IA

## Content of the repository

This repository contains 5 main files for the application:
- `saillance.py`: contains all the code necessary to run the saliency detection on an image.
- `points_interets.py`: contains all the code necessary to extract interest points from saliency map.
- `recadrage.py`: contains all the code used to crop an image based on a provided ratio and saliency informations (saliency map or interest points).
- `aesthetics.py`: contains all the code necessary to run the aesthetics scoring on an image.
- `smart_crop.py`: the main file of the application that combines all the steps and ensures the production of the final result using user input  
  
The repository also contains:
- `requirements.txt`: specifies all the required python libraries for the application to work
- `detection_saillance`: contains a dataset download script, models, training and testing programs used for the saliency detection neural network
- `aesthetics_pred`: contains a dataset download script, models, training and testing programs used for the aesthetics scoring neural network

## Install dependencies

Start by installing the required python packages using:
```
python3 -m pip install -r requirements.txt
```

This can be executed in a venv in order to limit these packages to a venv.

## Retrieve weights

In order for the algorithm to work, one must first retrieve the weights of the saliency detection neural network and the aesthetics scoring neural network.
The weights are available [here](https://github.com/LeoMarche/ProjetIA/releases/download/poc/premade.pth) and here (TODO upload and insert link).

## How to use

Once the dependencies installed, run `python3 smart_crop.py <path_to_saliency_weights> <path_to_aesthetics_weights> <path_to_image> <resize_ratio>`  
It will make windows appear to show intermediate results, close them to continue the algorithm

Example: `python3 smart_crop.py model1.pth model2.pth imgtest1.jpg 1.5` in order to run the algorithm and to intelligently crop the imgtest1.jpg image at a 1.5 ratio.

## Result

The final results of the application are created in a folder called `output`  
It contains each cropping suggested by the program using a specific name format:  
`output - <datetime> - choix<suggestion number> - <aesthetics score> .jpg`