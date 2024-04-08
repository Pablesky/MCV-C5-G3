# TASK IMAGE2TEXT

from torchvision.models import resnet152, ResNet152_Weights
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
import torch.optim as optim
from torch.optim import lr_scheduler
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from utils_w4 import *
import pickle
import wandb
import time
from transformers import BertModel, BertTokenizer
wandb.login(key='e6655e50ed903faab6661dc405cd17a07bbf8914')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        
        weights = ResNet152_Weights.DEFAULT
        self.model = resnet152(weights=weights).to(device)
        self.model.fc = nn.Identity()

    def forward(self, x):
        output = self.model(x)
        return output
    
class Bert2Res(nn.Module):
    def __init__(self, n_input = 768, n_ouput = 2048):
        super(Bert2Res, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=n_input, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=n_ouput)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
def train(model_image, model_text, loss_func, mining_func, device, train_loader, optimizer_image, optimizer_text, epoch):
    model_image.train()
    model_text.train()

    for batch_idx, (images, captions, text_features, ids) in enumerate(train_loader):
        images = images.to(device)
        text_features = text_features.to(device)
        
        # Image Embedding
        image_features = model_image(images)
        
        # Text Embedding
        text_features = model_text(text_features)

        optimizer_image.zero_grad()
        optimizer_text.zero_grad()

        # embeddings es don treiem els anchors
        # ref_emb es don treiem les positive and negative

        indices_tuple = mining_func(
            embeddings = text_features, 
            labels = ids, 
            ref_emb = image_features, 
            ref_labels = ids,
            )

        #loss = loss_func(embeddings, labels, indices_tuple)
        loss = loss_func(embeddings = text_features, indices_tuple=indices_tuple, ref_emb=image_features, labels = ids)
        loss.backward()

        optimizer_image.step()
        optimizer_text.step()

        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}".format(
                    epoch, batch_idx, loss
                )
            )

        wandb.log({"loss_train": loss})
"""
sweep_config = {
    'method': 'bayes',
    'metric': {'goal': 'minimize', 'name': 'loss_train'},
    'parameters': {

        'lr': {
            'distribution': 'uniform',
            'max': 0.0001,
            'min': 0.00001
        },
        'n_epochs': {
            'values':[1]

        },
        'miner' : {
            'values': ['TripletMarginMiner', 'BatchHardMiner']
            },
        'loss': {
            'values': ['TripletMarginLoss', 'MarginLoss']
        },
        'distance': {
            'values': ['Cosine', 'LP', 'squared_l2']
        },
        'margin': {
            'values': [1, 5, 10, 50, 100]
        },

        'batch_size': {
            'values': [16, 32, 64]
        },
        'type_of_miner': {
            'values': ['easy', 'semihard', 'hard', 'all']
        }
    }
}
"""

sweep_config = {
    'method': 'bayes',
    'metric': {'goal': 'minimize', 'name': 'loss_train'},
    'parameters': {

        'lr': {
            'values': [0.00001029]
        },
        'n_epochs': {
            'values':[1]

        },
        'miner' : {
            'values': ['BatchHardMiner']
            },
        'loss': {
            'values': ['MarginLoss']
        },
        'distance': {
            'values': ['Cosine']
        },
        'margin': {
            'values': [0.4867]
        },

        'batch_size': {
            'values': [32]
        },
        'type_of_miner': {
            'values': ['hard']
        }
    }
}

def caller(config = None):
    with wandb.init(config=config):

        config = wandb.config

        lr = config.lr
        n_epochs = config.n_epochs
        miner = config.miner
        loss = config.loss
        distance = config.distance
        margin = config.margin
        batch_size = config.batch_size
        type_of_miner = config.type_of_miner

        wandb.run.name = f'./model_bert_b_20_{margin}_{lr}_{batch_size}'

        bert_model = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        preprocess = ResNet152_Weights.DEFAULT.IMAGENET1K_V2.transforms()

        start = time.time()
        train_dataloader, test_dataloader = load_data_bert(bert_model, tokenizer, batch_size, preprocess)
        print("Load dataset time: ", time.time()-start)

        model_image = EmbeddingNet().to(device)
        model_text = Bert2Res().to(device)


        for layer in model_text.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

        
        if distance == "Cosine":
            distance_ = distances.CosineSimilarity()
        elif distance == "squared_l2":
            distance_ = distances.LpDistance(p=2)
        else:
            distance_ = distances.LpDistance()

        reducer = reducers.ThresholdReducer(low=0)

        #reducer = reducers.AvgNonZeroReducer()

        if loss == "TripletMarginLoss":
            loss_fn = losses.TripletMarginLoss(margin=margin, distance=distance_, reducer=reducer)
        else:
            loss_fn = losses.MarginLoss(margin=margin, nu=0, beta=1.2, triplets_per_anchor="all", learn_beta=False, num_classes=None, distance = distance_)

        if miner == "TripletMarginMiner":
            miner = miners.TripletMarginMiner(
            margin=margin, distance=distance_, type_of_triplets=type_of_miner)
        else:
            miner = miners.BatchHardMiner()

        optimizer_image = optim.Adam(model_image.parameters(), lr=lr)
        optimizer_text = optim.Adam(model_text.parameters(), lr=lr)

        #optimizer_image = optim.NAdam(model_image.parameters(), lr=lr)
        #optimizer_text = optim.NAdam(model_text.parameters(), lr=lr)

        time_counter = []
        for epoch in range(1, n_epochs + 1):
            start = time.time()
            train(model_image, model_text, loss_fn, miner, device, train_dataloader, optimizer_image, optimizer_text, epoch)
            end = time.time()
            time_counter.append(end-start)

        torch.save(model_image.state_dict(), f'./model_image_bert_b_20.pth')
        torch.save(model_text.state_dict(), f'./model_text_bert_b_20.pth')

        #generate_embeddings(f'./model_image_bert_b_10.pth', f'./model_text_bert_b_10.pth', '_10_txt2img')
    
    print("Average time epoch: ", np.mean(time_counter))

if __name__ == '__main__':
        
    sweep_id = wandb.sweep(sweep_config, project="W4_c")

    wandb.agent(sweep_id, function=caller, count=1)