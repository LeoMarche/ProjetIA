# Projet IA

## Content of the repository

This repository contains three main files:
- `saillance.py`: contains all the code necessary to run the saliency detection on an image.
- `points_interets.py`: contains all the code necessary to extract interest points from saliency map.
- `recadrage.py`: contains all the code used to crop an image based on a provided ratio and saliency informations (saliency map or interest points).

## How to use

Start by installing the require python packages using:
```
python3 -m pip install -r requirements.txt
```

This can be executed in a venv in order to limit these packages to a venv.

Once the dependencies installed, run `python3 ./saillance.py ./detection_saillance/results/premade.pth ./imgtest1.jpg 1.5` in order to run the algorithm and to intelligently crop the imgtest1.jpg image at a 1.5 ratio.