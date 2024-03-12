from torchvision.models import resnet152, ResNet152_Weights
from torchinfo import summary
import torch
from PIL import Image
from torchvision.io import read_image
import os
from pathlib import Path
import pickle
import numpy as np
# This is the way to create a feature extractor from a model
# https://github.com/pytorch/vision/releases/tag/v0.11.0
from torchvision.models.feature_extraction import create_feature_extractor

HEIGHT = 256
WIDTH = 256
CHANNELS = 3
BATCH_SIZE = 1

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



if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    weights = ResNet152_Weights.DEFAULT
    
    preprocess = weights.IMAGENET1K_V2.transforms()
    # The images are resized to resize_size=[232] using interpolation=InterpolationMode.BILINEAR, followed by a central crop of crop_size=[224]. 
    # Finally the values are first rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    
    model = resnet152(weights=weights).to(device)

    # As we are not training the model
    model.eval()

    model2 = create_feature_extractor(model, return_nodes=return_nodes)

    dataset_features = []

    root_path = Path('../MIT_split/')

    with torch.inference_mode():

        for folder in root_path.iterdir():

            class_vectors = []
            class_labels = []

            for class_folder in folder.iterdir():

                for image_path in class_folder.glob('*.jpg'):
                    image = read_image(str(image_path)).to(device)
                    intermediate_outputs = model2(preprocess(image).unsqueeze(0))
                    output = intermediate_outputs['avgpool'].flatten().detach().cpu().numpy()

                    class_vectors.append(output)
                    class_labels.append(classes_translate[class_folder.name])

            dataset_features.append([class_vectors, class_labels])
                
    
    with open('datasetResnet152.pkl', 'wb') as f:
        pickle.dump(dataset_features, f)

                


