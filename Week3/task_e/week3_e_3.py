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
import cv2
import matplotlib.pyplot as plt
from utils import *
from sklearn.neighbors import NearestNeighbors

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

COCO_LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 
'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye' 'glasses', 
'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
'couch', 'potted plant', 'bed', 'mirror', 'dining' 'table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']


train_path = '/ghome/mcv/datasets/C5/COCO/train2014'
val_path = '/ghome/mcv/datasets/C5/COCO/val2014'
annotation_path = '/ghome/mcv/datasets/C5/COCO/mcv_image_retrieval_annotations.json'
with open(annotation_path, 'r') as f:
    annotations = json.load(f)

OPTIMIZER='nadam'
REGULARIZER = 0
EPOCHS = 1
IMG_HEIGHT=224
IMG_WIDTH = 224

###############
MARGIN = 10
VERSION = 'V2'
AGGREGATION = 'weighted'
LEARNING_RATE=0.00001
BATCH_SIZE= 16

###############

subset = 'database'
filename = f'embeddings_{subset}_{VERSION}_{MARGIN}_{AGGREGATION}_{LEARNING_RATE}_{BATCH_SIZE}.txt'
database_embeddings = np.loadtxt(filename)
database_labels = dict() 
for label in sorted(annotations[subset].keys()):
    for image in annotations[subset][label]:
        if image in database_labels:
            database_labels[image].append(label)
        else:
            database_labels[image] = [label]

subset = 'val'
filename = f'embeddings_{subset}_{VERSION}_{MARGIN}_{AGGREGATION}_{LEARNING_RATE}_{BATCH_SIZE}.txt'
validation_embeddings = np.loadtxt(filename)
validation_labels = dict() 

for label in sorted(annotations[subset].keys()):
    for image in annotations[subset][label]:
        if image in validation_labels:
            validation_labels[image].append(label)
        else:
            validation_labels[image] = [label]

subset = 'test'
filename = f'embeddings_{subset}_{VERSION}_{MARGIN}_{AGGREGATION}_{LEARNING_RATE}_{BATCH_SIZE}.txt'
test_embeddings = np.loadtxt(filename)
test_labels = dict() 

for label in sorted(annotations[subset].keys()):
    for image in annotations[subset][label]:
        if image in test_labels:
            test_labels[image].append(label)
        else:
            test_labels[image] = [label]

knn = NearestNeighbors(n_neighbors=9, metric = 'cosine')
knn.fit(database_embeddings)

distances, indices = knn.kneighbors(validation_embeddings, n_neighbors = 5)

results = np.zeros(5)

start = time.time()
for query_id, neighbors in enumerate(indices[300:310]):

    id_query = sorted(list(validation_labels.keys()))[query_id+300]
    labels_query = validation_labels[id_query]

    path_query = f'COCO_val2014_{str(id_query).zfill(12)}.jpg'
    img_query = cv2.imread(val_path + '/' + path_query, cv2.IMREAD_COLOR)

    name_labels= [COCO_LABELS[int(label)-1] for label in labels_query ]

    plt.figure(figsize=(25, 15))

    plt.subplot(1, 6, 1)
    plt.imshow(cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB))
    plt.title(f'{name_labels}')
    plt.axis('off')

    for idx, i in enumerate(neighbors):
        id_retrieved = sorted(list(database_labels.keys()))[i]
        labels_retrieved = database_labels[id_retrieved]

        if set(labels_query).intersection(set(labels_retrieved)):
            results[idx:] += 1
            #break   

        path_retrieved = f'COCO_train2014_{str(id_retrieved).zfill(12)}.jpg'
        img_retrieved = cv2.imread(train_path + '/' + path_retrieved, cv2.IMREAD_COLOR)

        name_labels= [COCO_LABELS[int(label)-1] for label in labels_retrieved ]

        plt.subplot(1, 6, idx+2)
        plt.imshow(cv2.cvtColor(img_retrieved, cv2.COLOR_BGR2RGB))
        plt.title(f'{name_labels}')
        plt.axis('off')

    plt.savefig(f'test{query_id+10}.png')
    plt.close()

print("Average time: ", time.time()-start)
print(f'{VERSION}_{MARGIN}_{AGGREGATION}_{LEARNING_RATE}_{BATCH_SIZE}')
print("Results:", results/len(validation_embeddings))

