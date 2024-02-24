# Week 1: Introduction to Pytorch - Image classification

## Folder structure 
The code and data is structured as follows:

        .
        ├── OriginalKerasModel.py        # Original image classification model in Keras
        ├── Week1.py                     # Image classification model in Pytorch
        └── ModelLayersKeras.txt         # Summary of the Keras model

## Requirements
Standard Computer Vision python packages are used. Regarding the python version, Python >= 3.6 is needed.

- PyTorch:
  ```pip install torch```
- TorchVision:
  ```pip install torchvision```
- TorchInfo:
  ```pip install torchinfo```
- WandB:
  ```pip install wandb```


## Tasks
The main goal of this project is to get familiarized with the PyTorch framework. Thus, the main tasks are:

- Understand Pytorch framework.
- Implement Image Classification network from C3 in Pytorch.
- Compute loss graphs and compare them with yours from Keras.
- Compute accuracy graphs and compare them with yours from Keras.

All the hyperparameters are optimized using wandb.ai.
