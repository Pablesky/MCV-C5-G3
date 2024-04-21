import os, sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import torchinfo # to print model summary
from torchinfo import summary # to print model summary
from tqdm.auto import tqdm # used in train function
import torchvision # print model image
from torchview import draw_graph # print model image
import random
from PIL import Image
import glob
from pathlib import Path
from timeit import default_timer as timer  
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet152, ResNet152_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


print("torch version: ", torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("devide: ", device)


def walk_through_dir(data_path):
  for dirpath, dirnames, filenames in os.walk(data_path):
    filenames = [f for f in filenames if not f[0] == '.'] # to exclude the ".DS_Store"
    dirnames[:] = [d for d in dirnames if not d[0] == '.'] # to exclude the ".DS_Store"
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        
        weights = ResNet152_Weights.DEFAULT
        self.model = resnet152(weights=weights).to(device)
        self.model.fc = nn.Identity()

    def forward(self, x):
        output = self.model(x)
        return output


def transform_data():
    weights = ResNet152_Weights.DEFAULT
    preprocess = weights.IMAGENET1K_V2.transforms()

    return preprocess


def loadImageData(train_dir,valid_dir,test_dir,data_train_transform, data_valid_test_transform):
    # Creating training set
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                    transform=data_train_transform, # transforms to perform on data (images)
                                    target_transform=None) # transforms to perform on labels (if necessary)

    # Creating validation set
    valid_data = datasets.ImageFolder(root=valid_dir, # target folder of images
                                    transform=data_valid_test_transform, # transforms to perform on data (images)
                                    target_transform=None) # transforms to perform on labels (if necessary)

    #Creating test set
    test_data = datasets.ImageFolder(root=test_dir, transform=data_valid_test_transform)

    print(f"Train data:\n{train_data}\nValidation data:\n{valid_data}\nTest data:\n{test_data}")

    # Get class names as a list
    class_names = train_data.classes
    print("Class names: ",class_names)

    # Check the lengths
    print("The lengths of the training, validation and test sets: ", len(train_data), len(valid_data), len(test_data))  

    return train_data, valid_data, test_data, class_names


def myDataLoader(train_data, valid_data, test_data, NUM_WORKERS, BATCH_SIZE, BATCH_SIZE_VALID, BATCH_SIZE_TEST):

    # Turn train and test Datasets into DataLoaders
    train_dataloader = DataLoader(dataset=train_data, 
                                batch_size=BATCH_SIZE, # how many samples per batch?
                                num_workers=NUM_WORKERS,
                                shuffle=True) # shuffle the data?

    # Turn train and test Datasets into DataLoaders
    valid_dataloader = DataLoader(dataset=valid_data, 
                                batch_size=BATCH_SIZE_VALID, # how many samples per batch?
                                num_workers=NUM_WORKERS,
                                shuffle=True) # shuffle the data?

    test_dataloader = DataLoader(dataset=test_data, 
                                batch_size=BATCH_SIZE_TEST, 
                                num_workers=NUM_WORKERS,
                                shuffle=False) # don't usually need to shuffle testing data

    # Now let's get a batch image and check the shape of this batch.    
    img, label = next(iter(train_dataloader))

    # Note that batch size will now be 1.  
    print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    print(f"Label shape: {label.shape}")

    return train_dataloader, valid_dataloader, test_dataloader  


def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    loss_epoch = []
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        print(
            "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                epoch, batch_idx, loss, mining_func.num_triplets
            )
        )
        loss_epoch.append(loss.cpu().detach().numpy())

    return loss_epoch

def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


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

def get_data_sets_path(data_path):
    train_dir = os.path.join(data_path,"train")
    valid_dir = os.path.join(data_path,"valid")
    test_dir = os.path.join(data_path,"test")
    print("train dir: ", train_dir)
    print("valid dir: ", valid_dir)
    print("test dir: ", test_dir)
    return train_dir, valid_dir, test_dir

def flatten(lst):
    flattened_list = []
    for sublist in lst:
        if isinstance(sublist, list):
            flattened_list.extend(flatten(sublist))
        else:
            flattened_list.append(sublist)
    return flattened_list


def main():
    device = torch.device("cuda")

    transform = transform_data()

    batch_size = 128

    data_path = sys.argv[1]

    train_dir, valid_dir, test_dir = get_data_sets_path(data_path)

    train_data, valid_data, test_data, class_names = loadImageData(train_dir,valid_dir,test_dir,transform, transform)
    train_dataloader, valid_dataloader, test_dataloader = myDataLoader(
        train_data=train_data, 
        valid_data=valid_data, 
        test_data=test_data, 
        NUM_WORKERS=0, 
        BATCH_SIZE=batch_size, 
        BATCH_SIZE_VALID=batch_size, 
        BATCH_SIZE_TEST=1
    )


    model = EmbeddingNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 5

    ### pytorch-metric-learning stuff ###
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(
        margin=0.2, distance=distance, type_of_triplets="semihard"
    )
    ### pytorch-metric-learning stuff ###

    loss_whole_train = []
    for epoch in range(1, num_epochs + 1):
        loss_whole_train.append(train(model, loss_func, mining_func, device, train_dataloader, optimizer, epoch))
        # test(train_dataloader, test_dataloader, model, accuracy_calculator)

    metric_learning_path = "embedding_net"
    if not os.path.exists(metric_learning_path):
        os.makedirs(metric_learning_path)

    torch.save(model.state_dict(), os.path.join(metric_learning_path, 'embedding_net.pth'))

    with open('lista_loss.pkl', 'wb') as f:
        pickle.dump(loss_whole_train, f)

    flattened_list = flatten(loss_whole_train)
    plt.plot(flattened_list)
    plt.title("Loss over training")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(metric_learning_path, 'loss.png'))

if __name__ == "__main__":
    main()  