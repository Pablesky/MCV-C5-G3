import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2, fasterrcnn_resnet50_fpn
import os
import json
from PIL import Image
import cv2
import time

def getOptimizer(name, parameters, lr, regularization):
    if name == 'adam':
        return optim.Adam(parameters, lr=lr, weight_decay=regularization)
    elif name == 'adamw':
        return optim.AdamW(parameters, lr=lr, weight_decay=regularization)
    elif name == 'sgd':
        return optim.SGD(parameters, lr=lr, weight_decay=regularization)
    elif name == 'nadam':
        return optim.NAdam(parameters, lr=lr, weight_decay=regularization)
    elif name == 'adadelta':
        return optim.Adadelta(parameters, lr=lr, weight_decay=regularization)

class TripletCOCO(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    """
    def __init__(self, img_path, annotations, subset, transform=None):
        self.img_path = img_path
        self.annotations = annotations
        self.subset = subset
        self.transform = transform
        self.images_labels = dict() #dict {image_id: [label1, label2, ...]}

        for label in self.annotations[subset].keys():
            for image in self.annotations[subset][label]:
                if image in self.images_labels:
                    self.images_labels[image].append(label)
                else:
                    self.images_labels[image] = [label]


    def __getitem__(self, index):
        anchor_id = list(self.images_labels.keys())[index]
        anchor_labels = self.images_labels[anchor_id]
        anchor_name = f'COCO_{self.subset}2014_{str(anchor_id).zfill(12)}.jpg'

        positive_label = np.random.choice(anchor_labels)
        positive_id = np.random.choice(self.annotations[self.subset][positive_label])
        while positive_id == anchor_id:
            positive_label = np.random.choice(anchor_labels)
            positive_id = np.random.choice(self.annotations[self.subset][positive_label])
        positive_labels = self.images_labels[positive_id]
        positive_name = f'COCO_{self.subset}2014_{str(positive_id).zfill(12)}.jpg'

        negative_label = np.random.choice(list(set(list(self.annotations[self.subset].keys())) - set(anchor_labels+positive_labels)))
        negative_id = np.random.choice(self.annotations[self.subset][negative_label])
        negative_name = f'COCO_{self.subset}2014_{str(negative_id).zfill(12)}.jpg'

        img1 = cv2.imread(self.img_path + '/' + anchor_name, cv2.IMREAD_COLOR)
        img2 = cv2.imread(self.img_path + '/' + positive_name, cv2.IMREAD_COLOR)
        img3 = cv2.imread(self.img_path + '/' + negative_name, cv2.IMREAD_COLOR)
        
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        img3 = Image.fromarray(img3)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return (img1, img2, img3), []

    def __len__(self):
        return len(self.images_labels)

class EmbeddingNet(nn.Module):
    def __init__(self, version, aggregate):
        super(EmbeddingNet, self).__init__()

        def save_features(module, input, output):
            self.features.append(output)

        def save_scores(module, input, output):
            self.scores.append(output)

        def save_proposals(module, input, output):
            self.proposals.append(output)

        self.features = []
        self.scores = []
        self.proposals = []
        self.aggregate = aggregate

        if version == "V1":
            self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1, min_size = 224, max_size = 224)
            layer_features = self.model.roi_heads.box_head.fc7

        if version == "V2":
            self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, min_size = 224, max_size = 224)
            layer_features = self.model.roi_heads.box_head[6]

        layer_scores  = self.model.roi_heads.box_predictor.cls_score
        layer_proposals = self.model.rpn

        self.features_hook = layer_features.register_forward_hook(save_features)
        self.scores_hook = layer_scores.register_forward_hook(save_scores)
        self.proposals_hook = layer_proposals.register_forward_hook(save_proposals)

    def forward(self, x):
        
        if self.model.training:
            target = {}
            target["boxes"] = torch.zeros((0,4)).to(x.device)
            target["labels"] = torch.zeros((0), dtype = torch.int64).to(x.device)
            target["image_id"] = torch.zeros((0), dtype = torch.int64).to(x.device)
            
            targets = [target] * x.shape[0]
        
            self.model(x, targets)
        else:
            self.model(x)

        _scores = self.scores[0]
        del self.scores[0]

        _proposals = self.proposals[0]
        del self.proposals[0]

        _features = self.features[0]
        del self.features[0]
        

        if self.model.training:
            proposals_per_image = 512
        else:
            n_images = range(len(x))
            proposals_per_image = [len(_proposals[0][i]) for i in n_images]


        features_per_image = _features.split(proposals_per_image, 0)
        scores_per_image = _scores.split(proposals_per_image, 0)


        if self.aggregate == 'mean':
            features = [tensor.mean(dim=0) for tensor in features_per_image]

        elif self.aggregate == 'weighted':
            proposal_prob_class = [torch.nn.functional.softmax(tensor, dim=1) for tensor in scores_per_image] #softmax by row

            max_proposal_prob_class = [torch.max(tensor, dim=1)[0] for tensor in proposal_prob_class] #max by row
            
            weighted_features = [tensor1 * tensor2.unsqueeze(1) for tensor1, tensor2 in zip(features_per_image, max_proposal_prob_class)]

            features = [torch.mean(tensor, dim = 0) for tensor in weighted_features]

        return torch.stack(features, dim=0)

    def get_embedding(self, x):
        return self.forward(x)

    def cleanup_hooks(self):
        if self.features_hook is not None:
            self.features_hook.remove()
        if self.scores_hook is not None:
            self.scores_hook.remove()
        if self.proposals_hook is not None:
            self.proposals_hook.remove()

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

def compute_image_embeddings(model, img_path, annotations, subset, transform, filename):
    images_labels = dict() 

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    for label in sorted(annotations[subset].keys()):
        for image in annotations[subset][label]:
            if image in images_labels:
                images_labels[image].append(label)
            else:
                images_labels[image] = [label]

    if subset == 'train' or subset == 'database':
        subset_image = 'train'
    else:
        subset_image = 'val'

    time_counter = []
    with torch.no_grad():
        for image_id in sorted(list(images_labels.keys())):
            
            image_name = f'COCO_{subset_image}2014_{str(image_id).zfill(12)}.jpg'
            image = cv2.imread(img_path + '/' + image_name, cv2.IMREAD_COLOR)

            image = Image.fromarray(image)
            image = transform(image).to(device)
            image = torch.stack([image])

            start = time.time()
            output = model(image)
            end = time.time()

            np.savetxt(filename, output.cpu().numpy())

            time_counter.append(end-start)


    print("Average time: ", np.mean(time_counter))

    filename.close()