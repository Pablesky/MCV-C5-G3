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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = torch.cuda.is_available()




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



def caller():

    lr = 0.00001802
    n_epochs = 5
    miner = 'BatchHardMiner'
    loss = 'TripletMarginLoss'
    distance = 'LP'
    margin = 0.9821
    batch_size = 64
    type_of_miner = 'semihard'

    root_dir = '../MIT_split/'
    train_dir = root_dir + 'train'
    test_dir = root_dir + 'test'

    train_dataset, test_dataset, train_dataloader, test_dataloader = load_data(train_dir, test_dir, batch_size)

    model = EmbeddingNet().to(device)
    
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

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        train(model, loss_fn, miner, device, train_dataloader, optimizer, epoch)

    test(train_dataset, test_dataset, model, accuracy_calculator)

    torch.save(model.state_dict(), 'bestModelTriplet.pth')


if __name__ == '__main__':
    caller()



    


    