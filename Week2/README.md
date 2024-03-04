# Week 2: Object detection, recognition and segmentation

## Folder structure 
The code and data is structured as follows:

        .
        project_name/
        ├── kitti2coco.py                 # Function to transform Kitti annotations to COCO format
        ├── kitti2yolo.py                 # Function to transform Kitti annotations to YOLO format
        ├── task_c_1.py                   # Inference on pretrained Faster R-CNN
        ├── task_c_2.py                   # Inference on pretrained Mask R-CNN
        ├── task_d_1.py                   # Evaluation of pretrained Faster R-CNN
        ├── task_d_2.py                   # Evaluation of pretrained Mask R-CNN
        ├── task_e_1.py                   # Fine-tuning of pretrained Faster R-CNN
        ├── task_e_1_optimization.py      # Fine-tuning of pretrained Faster R-CNN (W&B version)
        ├── task_e_2.py                   # Fine-tuning of pretrained Mask R-CNN
        ├── task_e_2_optimization.py      # Fine-tuning of pretrained Mask R-CNN (W&B version)
        ├── task_f.py                     # Inference and evaluation of YOLOv9
        └── visualizationAnnotations.py   # A file to see the results of the kitti2coco.py functions


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
- Detectron2


## Tasks
The main goal of this project is to get familiarized in object detection, recognition and segmentation, specifically in the Detectron2 framework. Thus, the main tasks are:

- c) Run inference with pre-trained Faster R-CNN (detection) and Mask R-CNN (detection and segmentation) on KITTI-MOTS dataset.
- d) Evaluate pre-trained Faster R-CNN (detection) and Mask R-CNN (detection and segmentation) on KITTI-MOTS dataset.
- e) Fine-tune Faster R-CNN and Mask R-CNN on KITTI-MOTS.
- f) Apply some other object detection model (YOLO) to KITTI-MOTS.


All the hyperparameters are optimized using wandb.ai.
