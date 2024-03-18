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
wandb.login(key='e6655e50ed903faab6661dc405cd17a07bbf8914')

###############
MARGIN = 100
VERSION = 'V1'
AGGREGATION = 'mean'
LEARNING_RATE=0.00001
BATCH_SIZE= 64

for VERSION in ['V1']:
    writer = wandb.init(project='C5_W3', job_type='train',
    config={
        'LEARNING_RATE': str(LEARNING_RATE),
        'MARGIN': str(MARGIN),
        'VERSION': VERSION,
        'AGGREGATION' : AGGREGATION, 
        'BATCH_SIZE' : str(BATCH_SIZE)
        })

    wandb.run.name = f'model_trained_{VERSION}_{MARGIN}_{AGGREGATION}_{LEARNING_RATE}_{BATCH_SIZE}_2'

    if VERSION == 'V1':
        transform = transforms.Compose([
            FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms(),
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH))])
    else:
        transform = transforms.Compose([
            FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms(),
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH))])
        
    dataset = TripletCOCO(train_path, annotations, 'train', transform)
    train_dataset = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = EmbeddingNet(VERSION, AGGREGATION)
    model.to(device)
    optimizer = getOptimizer(OPTIMIZER, model.parameters(), LEARNING_RATE, REGULARIZER)

    criterion = TripletLoss(margin = MARGIN)

    model.train()

    count = 0
    time_counter = []

    for epoch in range(EPOCHS):
        start = time.time()

        for i, data in enumerate(train_dataset):
            optimizer.zero_grad()

            inputs, labels = data
            x1, x2, x3 = inputs[0].to(device), inputs[1].to(device), inputs[2].to(device)
        
            output1 = model(x1)
            output2 = model(x2)
            output3 = model(x3)

            loss = criterion(output1, output2, output3)

            loss.backward()
            optimizer.step()

            count += 1
            wandb.log({"iteration": count, "loss": loss.item()})
        end = time.time()
        time_counter.append(end-start)
        wandb.finish()
        torch.save(model.state_dict(), f'./model_trained_{VERSION}_{MARGIN}_{AGGREGATION}_{LEARNING_RATE}_{BATCH_SIZE}.pth')
print("Average time epoch: ", np.mean(time_counter))
