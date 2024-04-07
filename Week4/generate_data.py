from utils import *
import torch 
from torchvision.models import ResNet152_Weights
import pickle

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weights = ResNet152_Weights.DEFAULT

    preprocess = weights.IMAGENET1K_V2.transforms()

    train_dataset = COCODataset(
        root = '../COCO/train2014', 
        json_path = '../COCO/captions_train2014.json', 
        transform = preprocess)

    val_dataset = COCODataset(
        root = '../COCO/val2014', 
        json_path = '../COCO/captions_val2014.json', 
        transform = preprocess)
    
    with open('val_dataset_COCO.pkl', 'wb') as f:
        pickle.dump(val_dataset, f)

    with open('train_dataset_COCO.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)

