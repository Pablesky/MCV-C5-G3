import torch.nn as nn
import torch
from torchvision.models import resnet152, ResNet152_Weights
from transformers import BertTokenizer, BertModel
from torchvision.io import ImageReadMode
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import pandas as pd
import os
import numpy as np
import pickle

import matplotlib

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        
        self.weights = ResNet152_Weights.DEFAULT
        self.model_image = resnet152(weights=self.weights)
        self.model_image.fc = nn.Identity()
        self.model_image = self.model_image.to(device)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model_text = BertModel.from_pretrained('bert-base-uncased')

        self.model_audio = torch.hub.load('harritaylor/torchvggish', 'vggish')

        self.model_image.eval()
        self.model_text.eval()
        self.model_audio.eval()

        self.image_2_emb = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        ).to(device)

        self.text_2_emb = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        ).to(device)

        self.audio_2_emb = nn.Sequential(
            nn.Linear(128 * 15, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU()
        ).to(device)

        self.classifier = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        ).to(device)

    def forward(self, image, text, audio):
        '''
        image -> should be an image tensor
        text -> should be a list of strings
        '''
        encoding = self.tokenizer.batch_encode_plus(
                text,                    # List of input texts
                padding=True,              # Pad to the maximum sequence length
                truncation=True,           # Truncate to the maximum sequence length if necessary
                return_tensors='pt',      # Return PyTorch tensors
                add_special_tokens=True    # Add special tokens CLS and SEP
            )
        input_ids = encoding['input_ids'] 
        attention_mask = encoding['attention_mask']
        
        outputs = self.model_text(input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state

        text_features = word_embeddings.mean(dim=1)
        text_features = text_features.mean(dim=0)

        image_features = self.model_image(image)

        audio_features = self.model_audio(audio)


        text_features = text_features.reshape(1, -1).to(device)
        audio_features = audio_features.reshape(1, -1).to(device)
        image_features = image_features.reshape(1, -1).to(device)

        image_features = self.image_2_emb(image_features)
        text_features = self.text_2_emb(text_features)
        audio_features = self.audio_2_emb(audio_features)

        contenation = torch.cat((image_features, text_features, audio_features), dim=1)

        output = self.classifier(contenation)

        return output

class MyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        row = self.files.iloc[idx]

        video_name = row['VideoName'][:-4 ]
        userID = row['UserID']
        age_group = row['AgeGroup']
        gender = row['Gender']
        ethnicity = row['Ethnicity']

        base_path = self.root_dir + '/' + str(age_group) + '/' + video_name
        noise_path = base_path + '_noise'

        img_path = base_path + '.jpg'
        aud_path = base_path + '.wav'
        txt_path = base_path + '.pkl'

        image = read_image(img_path, ImageReadMode.RGB)
        image = self.transform(image)

        with open(txt_path, 'rb') as f:
            text = pickle.load(f)

        return (image, text, aud_path), age_group
    
def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10, eval_interval=20):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, ((images, texts, audios), labels) in enumerate(train_loader):
            total_outputs = []
            optimizer.zero_grad()
            for idx in range(images.size(0)):  # Iterate over each sample in the batch
                image = images[idx].unsqueeze(0).to(device)
                text = texts[idx]
                audio = audios[idx]
                label = labels[idx]

                print(image.shape, text, audio, label)

                outputs = model(image, text, audio)
                total_outputs.append(outputs)
            
            total_outputs = torch.cat(total_outputs, dim=0).to(device)
            loss = criterion(total_outputs, labels.to(device))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, batch_idx + 1, running_loss / 100))
            running_loss = 0.0

if __name__ == '__main__':

    batch_size = 64

    mydataset_train = MyDataset(
        csv_file='data/train_set_age_labels_noise.csv',
        root_dir='data/train',
        transform=ResNet152_Weights.DEFAULT.IMAGENET1K_V2.transforms()
    )

    mydataset_test = MyDataset(
        csv_file='data/test_set_age_labels.csv',
        root_dir='data/test',
        transform=ResNet152_Weights.DEFAULT.IMAGENET1K_V2.transforms()
    )

    mydataset_val = MyDataset(
        csv_file='data/valid_set_age_labels.csv',
        root_dir='data/valid',
        transform=ResNet152_Weights.DEFAULT.IMAGENET1K_V2.transforms()
    )

    dataloader_train = DataLoader(
        mydataset_train, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=0)
    
    dataloader_test = DataLoader(
        mydataset_test, 
        batch_size=1,
        shuffle=False, 
        num_workers=0)
    
    dataloader_val = DataLoader(
        mydataset_val, 
        batch_size=1,
        shuffle=False, 
        num_workers=0)

    transform = ResNet152_Weights.DEFAULT.IMAGENET1K_V2.transforms()
    model = MultimodalModel()

    for i in model.model_audio.parameters():
        i.requires_grad = False

    for i in model.model_text.parameters():
        i.requires_grad = False
    
    for i in model.model_image.parameters():
        i.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, dataloader_train, dataloader_test, criterion, optimizer, device, num_epochs=10, eval_interval=1)







