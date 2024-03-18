import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.io as io
import torchvision.transforms as transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2, fasterrcnn_resnet50_fpn
import os
import json
import wandb

from utils import *


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

annotation_path = '/ghome/mcv/datasets/C5/COCO/mcv_image_retrieval_annotations.json'
with open(annotation_path, 'r') as f:
    annotations = json.load(f)

OPTIMIZER='nadam'
REGULARIZER = 0
EPOCHS = 1
IMG_HEIGHT=224
IMG_WIDTH = 224

###############
MARGIN = 100
VERSION = 'V1'
AGGREGATION = 'mean'
LEARNING_RATE=0.00001
BATCH_SIZE= 64

###############
if VERSION == 'V1':
    transform = transforms.Compose([
        FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms(),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH))])
else:
    transform = transforms.Compose([
        FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms(),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH))])

model_load = EmbeddingNet(VERSION, AGGREGATION)
model_load.load_state_dict(torch.load(f'./model_trained_{VERSION}_{MARGIN}_{AGGREGATION}_{LEARNING_RATE}_{BATCH_SIZE}.pth'))
model_load.to(device)
model_load.eval()


img_path = '/ghome/mcv/datasets/C5/COCO/train2014'
subset = 'database'
filename = open(f'embeddings_{subset}_{VERSION}_{MARGIN}_{AGGREGATION}_{LEARNING_RATE}_{BATCH_SIZE}.txt', "wb")
compute_image_embeddings(model_load, img_path, annotations, subset, transform, filename)

img_path = '/ghome/mcv/datasets/C5/COCO/val2014'
subset = 'val'
filename = open(f'embeddings_{subset}_{VERSION}_{MARGIN}_{AGGREGATION}_{LEARNING_RATE}_{BATCH_SIZE}.txt', "wb")
compute_image_embeddings(model_load, img_path, annotations, subset, transform, filename)

img_path = '/ghome/mcv/datasets/C5/COCO/val2014'
subset = 'test'
filename = open(f'embeddings_{subset}_{VERSION}_{MARGIN}_{AGGREGATION}_{LEARNING_RATE}_{BATCH_SIZE}.txt', "wb")
compute_image_embeddings(model_load, img_path, annotations, subset, transform, filename)