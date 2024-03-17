from torchvision.models import resnet152, ResNet152_Weights
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset

import torch

import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor

from pathlib import Path

import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = torch.cuda.is_available()

return_nodes = {
    "avgpool" : "avgpool"
}

classes_translate = {
    'coast': 0,
    'forest': 1,
    'highway': 2,
    'inside_city': 3,
    'mountain': 4,
    'Opencountry': 5,
    'street': 6,
    'tallbuilding': 7
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
    

    
if __name__ == '__main__':
    weights = ResNet152_Weights.DEFAULT
    
    preprocess = weights.IMAGENET1K_V2.transforms()
    # The images are resized to resize_size=[232] using interpolation=InterpolationMode.BILINEAR, followed by a central crop of crop_size=[224]. 
    # Finally the values are first rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    
    model = EmbeddingNet()
    model.load_state_dict(torch.load('bestModelTriplet.pth'))
    model.to(device)

    # As we are not training the model
    model.eval()

    dataset_features = []

    root_path = Path('../MIT_split/')

    paths = []

    with torch.inference_mode():

        for folder in root_path.iterdir():

            class_vectors = []
            class_labels = []

            for class_folder in folder.iterdir():

                for image_path in class_folder.glob('*.jpg'):
                    image = read_image(str(image_path)).to(device)
                    paths.append(str(image_path))
                    intermediate_outputs = model(preprocess(image).unsqueeze(0))

                    class_vectors.append(intermediate_outputs.cpu()[0])
                    class_labels.append(classes_translate[class_folder.name])

            dataset_features.append([class_vectors, class_labels])
                
    
    with open('tripletPredictions.pkl', 'wb') as f:
        pickle.dump(dataset_features, f)

    with open('pathsTriplet.pkl', 'wb') as f:
        pickle.dump(paths, f)
    
