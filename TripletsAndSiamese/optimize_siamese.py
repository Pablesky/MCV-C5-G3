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

import wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = torch.cuda.is_available()

# # Dataset


def load_data(train_dir, test_dir, batch_size=8):
    weights = ResNet152_Weights.DEFAULT
    
    transformTrain = weights.IMAGENET1K_V2.transforms()
    transformTest = weights.IMAGENET1K_V2.transforms()
    
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=transformTrain, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

    test_data = datasets.ImageFolder(root=test_dir, 
                                    transform=transformTest)
    
    train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=batch_size, # how many samples per batch?
                              num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?

    test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=batch_size, 
                             num_workers=1, 
                             shuffle=False) # don't usually need to shuffle testing data

    return train_data, test_data, train_dataloader, test_dataloader


# # Models


return_nodes = {
    "avgpool" : "avgpool"
}

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        
        weights = ResNet152_Weights.DEFAULT
        self.model = resnet152(weights=weights).to(device)
        self.model2 = create_feature_extractor(self.model, return_nodes=return_nodes)

    def forward(self, x):
        intermediate_outputs = self.model2(x)
        output = intermediate_outputs['avgpool']
        # Get the sizes of the first two dimensions

        size_1, size_2 = output.size(0), output.size(1)

        # Reshape the tensor with flexible dimensions
        output = output.view(size_1, size_2)
        
        return output
    
# # Train

def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)

        indices_tuple = mining_func(embeddings, labels)

        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}".format(
                    epoch, batch_idx, loss
                )
            )
        wandb.log({"loss_train": loss})

def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels, False
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))

    wandb.log({"accuracy": accuracies["precision_at_1"]})


sweep_config = {
    'method': 'random',
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
        'miner_conf_positive' : {
            'values': ['hard', 'semihard', 'easy']
            },

        'miner_conf_negative' : {
            'values': ['hard', 'semihard', 'easy']
            },

        'loss': {
            'values': ["Contrastive", "NTX"]

        },
        'distance': {
            'values': ['Cosine', 'LP']
        },
        'p_margin': {
            'distribution': 'uniform',
            'max': 1,
            'min': 0
        },
        'n_margin': {
            'distribution': 'uniform',
            'max': 1,
            'min': 0
        },

        'batch_size': {
            'values': [16, 32, 64]
        }
    }
}



def caller(config = None):
    with wandb.init(config=config):

        config = wandb.config

        if not (config.miner_conf_positive == 'semihard' and config.miner_conf_negative == 'semihard'):
            lr = config.lr
            n_epochs = config.n_epochs
            miner_positive = config.miner_conf_positive
            miner_negative = config.miner_conf_negative
            loss = config.loss
            distance = config.distance
            p_margin = config.p_margin
            n_margin = config.n_margin
            batch_size = config.batch_size

            root_dir = '../MIT_split/'
            train_dir = root_dir + 'train'
            test_dir = root_dir + 'test'

            train_dataset, test_dataset, train_dataloader, test_dataloader = load_data(train_dir, test_dir, batch_size)

            model = EmbeddingNet().to(device)
            
            if distance == "Cosine":
                distance_ = distances.CosineSimilarity()
            else:
                distance_ = distances.LpDistance()

            if loss == "Contrastive":
                loss_fn = losses.ContrastiveLoss(pos_margin=p_margin, neg_margin=n_margin, distance = distance_)
            else:
                loss_fn = losses.NTXentLoss(temperature=0.1, distance = distance_)


            miner = miners.BatchEasyHardMiner(
                pos_strategy = miner_positive,
                neg_strategy = miner_negative
            )

            
            accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

            optimizer = optim.Adam(model.parameters(), lr=lr)

            for epoch in range(1, n_epochs + 1):
                train(model, loss_fn, miner, device, train_dataloader, optimizer, epoch)

            test(train_dataset, test_dataset, model, accuracy_calculator)


if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project="siamese")
    wandb.agent(sweep_id, function=caller, count=200)


    

