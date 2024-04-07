# TASK TEXT2IMAGE

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
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from utils import load_data, clean_sentence
import pickle
import wandb
import fasttext

fastModel = fasttext.load_model("fasttext_wiki.en.bin")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = torch.cuda.is_available()

# MODELS

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        
        weights = ResNet152_Weights.DEFAULT
        self.model = resnet152(weights=weights).to(device)
        self.model.fc = nn.Identity()

    def forward(self, x):
        output = self.model(x)
        return output
    
    
class Fast2Res(nn.Module):
    def __init__(self, n_input = 300, n_ouput = 2048):
        super(Fast2Res, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=n_input, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
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
        optimizer_image.zero_grad()
        optimizer_text.zero_grad()
        
        images = images.to(device)
        text_features = text_features.to(device)
        
        # Image Embedding
        image_features = model_image(images)
        
        # Text Embedding
        text_features = model_text(text_features)

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
            'values':[5, 6, 7, 8, 9, 10]

        },
        'miner' : {
            'values': ['TripletMarginMiner', 'BatchHardMiner']
            },
        'loss': {
            'values': ['TripletMarginLoss', 'MarginLoss']
        },
        'distance': {
            'values': ['Cosine', 'LP']
        },
        'margin': {
            'distribution': 'uniform',
            'max': 1,
            'min': 0
        },

        'batch_size': {
            'values': [16, 32, 64]
        },
        'type_of_miner': {
            'values': ['easy', 'semihard', 'hard', 'all']
        }
    }
}

def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


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

        train_dataset, test_dataset, train_dataloader, test_dataloader = load_data(fastModel, batch_size)

        model_image = EmbeddingNet().to(device)
        model_text = Fast2Res().to(device)

        model_image.train()
        model_text.train()

        for layer in model_text.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

        
        if distance == "Cosine":
            distance_ = distances.CosineSimilarity()
        else:
            distance_ = distances.LpDistance()

        reducer = reducers.ThresholdReducer(low=0)

        if loss == "TripletMarginLoss":
            loss_fn = losses.TripletMarginLoss(margin=margin, distance=distance_, reducer=reducer)
        else:
            loss_fn = losses.MarginLoss(margin=margin, nu=0, beta=1.2, triplets_per_anchor="all", learn_beta=False, num_classes=None, distance = distance_)

        if miner == "TripletMarginMiner":
            miner = miners.TripletMarginMiner(
            margin=margin, distance=distance_, type_of_triplets=type_of_miner)
        else:
            miner = miners.BatchHardMiner()

        
        accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

        optimizer_image = optim.Adam(model_image.parameters(), lr=lr)
        optimizer_text = optim.Adam(model_text.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):
            train(model_image, model_text, loss_fn, miner, device, train_dataloader, optimizer_image, optimizer_text, epoch)

        sufix = f'lr_{lr}_n_epochs_{n_epochs}_miner_{miner}_loss_{loss}_distance_{distance}_margin_{margin}_batch_size_{batch_size}_type_of_miner_{type_of_miner}'
        # QUITAR SI NO QUIERES GUARDAR
        torch.save(model_image.state_dict(), f'model_image_text2img_{sufix}.pth')
        torch.save(model_text.state_dict(), f'model_text_text2img_{sufix}.pth')
    
if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project="Text2Image-v0")
    wandb.agent(sweep_id, function=caller, count=100)