from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.io import read_image
import os
import json
import pickle
import fasttext
from torchvision.io import ImageReadMode
import torch.nn as nn
import torch
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = torch.cuda.is_available()

def plot_texts(query_image, query_index, captions, real_caption, path):
    f, axes = plt.subplots(5,2, figsize=(15, 15))

    # Plot input image
    axes[2, 0].imshow(query_image)
    axes[2, 0].axis('off')
    axes[3, 0].text(0, 0.95, real_caption, fontsize=24, wrap=True)
    axes[3, 0].axis('off')


    # Plot similar images
    for i, caption in enumerate(captions):
        axes[i, 1].text(0, 0.95, caption, fontsize=24, wrap=True)
        axes[i, 1].axis('off')

    axes[0, 0].set_visible(False)
    axes[1, 0].set_visible(False)
    axes[4, 0].set_visible(False)
    
    plt.tight_layout()
    # plt.show()

    query_index = str(query_index).zfill(12)

    f.savefig(f'{path}/retrieval_{query_index}.png')
    plt.close(f)

def plot_texts_custom(query_image, name, captions, path):
    f, axes = plt.subplots(5,2, figsize=(15, 15))

    # Plot input image
    axes[2, 0].imshow(query_image)
    axes[2, 0].axis('off')


    # Plot similar images
    for i, caption in enumerate(captions):
        axes[i, 1].text(0, 0.95, caption, fontsize=24, wrap=True)
        axes[i, 1].axis('off')

    axes[0, 0].set_visible(False)
    axes[1, 0].set_visible(False)
    axes[3, 0].set_visible(False)
    axes[4, 0].set_visible(False)
    
    plt.tight_layout()
    # plt.show()

    f.savefig(f'{path}/retrieval_{name}.png')
    plt.close(f)

def plot_image(query_text, query_index, images, real_image, path):
    f, axes = plt.subplots(5,2, figsize=(15, 15))

    # Plot input image
    axes[2, 0].text(0, 0.95, query_text, fontsize=24, wrap=True)
    axes[2, 0].axis('off')
    axes[3, 0].imshow(real_image)
    axes[3, 0].axis('off')


    # Plot similar images
    for i, image in enumerate(images):
        axes[i, 1].imshow(image)
        axes[i, 1].axis('off')

    axes[0, 0].set_visible(False)
    axes[1, 0].set_visible(False)
    axes[4, 0].set_visible(False)
    
    plt.tight_layout()
    # plt.show()

    query_index = str(query_index).zfill(12)

    f.savefig(f'{path}/retrieval_{query_index}.png')
    plt.close(f)

def plot_image_custom(query_text, name, images, path):
    f, axes = plt.subplots(5,2, figsize=(15, 15))

    # Plot input image
    axes[2, 0].text(0, 0.95, query_text, fontsize=24, wrap=True)
    axes[2, 0].axis('off')


    # Plot similar images
    for i, image in enumerate(images):
        axes[i, 1].imshow(image)
        axes[i, 1].axis('off')

    axes[0, 0].set_visible(False)
    axes[1, 0].set_visible(False)
    axes[3, 0].set_visible(False)
    axes[4, 0].set_visible(False)
    
    plt.tight_layout()
    # plt.show()

    f.savefig(f'{path}/retrieval_{name}.png')
    plt.close(f)

def reset_folder(path):
    try:
        shutil.rmtree(path)

    except:
        pass

    os.mkdir(path)

def clean_sentence(model, sentence):
    """
    Given a sentence, return the cleaned sentence with only the words inside the vocabulary.
    """
    words = sentence.lower().split()
    utils_words = [word for word in words if word in model]
    to_string = ' '.join(utils_words)

    return to_string

'''
Dataset class
torch.utils.data.Dataset is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override the following methods:

__len__ so that len(dataset) returns the size of the dataset.

__getitem__ to support the indexing such that dataset[i] can be used to get 
�
ith sample.

['caption']
{'image_id': 318556,
  'id': 48,
  'caption': 'A very clean and well decorated empty bathroom'}
--------------------------------------------------------------------------
['images']
 {'license': 5,
  'file_name': 'COCO_train2014_000000057870.jpg',
  'coco_url': 'http://images.cocodataset.org/train2014/COCO_train2014_000000057870.jpg',
  'height': 480,
  'width': 640,
  'date_captured': '2013-11-14 16:28:13',
  'flickr_url': 'http://farm4.staticflickr.com/3153/2970773875_164f0c0b83_z.jpg',
  'id': 57870}
'''
class COCODataset(Dataset):
    def __init__(self, root, json_path, transform):

        with open(json_path, 'r') as j:
            json_file = json.load(j)

        self.root = root
        self.image_path = []
        self.caption_text = []
        self.image_id = []
        self.transform = transform
        self.text_model = None

        for caption in json_file['annotations']:

            image_id = caption['image_id']
            caption_text = caption['caption']

            for image_file in json_file['images']:
                if image_file['id'] == image_id:

                    image_path = image_file['file_name']
                    self.image_path.append(image_path)
                    self.caption_text.append(caption_text)
                    self.image_id.append(image_id)

                    break
                    
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        actual_filename = self.image_path[idx]
        actual_filename = os.path.join(self.root, actual_filename)

        image = read_image(actual_filename, ImageReadMode.RGB)
        image = self.transform(image)

        id = self.image_id[idx]
        caption = self.caption_text[idx]

        return image, caption, self.text_model.get_sentence_vector(clean_sentence(self.text_model, caption)), id
    
    
def load_data(modelText, batch_size=8):


    with open('train_dataset_COCO.pkl', 'rb') as f:
        train_data = pickle.load(f)

    with open('val_dataset_COCO.pkl', 'rb') as f:
        val_data = pickle.load(f)

    train_data.text_model = modelText
    
    train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=batch_size, # how many samples per batch?
                              num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?
    
    val_data.text_model = modelText

    val_dataloader = DataLoader(dataset=val_data, 
                             batch_size=batch_size, 
                             num_workers=1, 
                             shuffle=False) # don't usually need to shuffle testing data
    
    return train_data, val_data, train_dataloader, val_dataloader

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

def generate_data(image_weights, text_weights, prefix):
    model_image = EmbeddingNet()
    model_text = Fast2Res()

    model_image.eval()
    model_text.eval()

    with torch.no_grad():

        if image_weights is not None:
            model_image.load_state_dict(torch.load(image_weights))

        if text_weights is not None:
            model_text.load_state_dict(torch.load(text_weights))

        model_image = model_image.to(device)
        model_text = model_text.to(device)

        with open('train_dataset_COCO.pkl', 'rb') as f:
            train_data = pickle.load(f)

        modelText = fasttext.load_model("fasttext_wiki.en.bin")
        train_data.text_model = modelText

        batch_size = 512
        
        train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=batch_size, # how many samples per batch?
                              shuffle=False) # shuffle the data?
        
        labels_train = []
        train_img_features = []
        train_text_features = []
        train_captions = []

        '''
        for images, captions, text_features, ids in tqdm(train_dataloader):
            images = images.to(device)
            text_features = torch.tensor(text_features).to(device)

            image_features = model_image(images)
            text_features = model_text(text_features)

            image_features = image_features.cpu().detach().numpy()
            text_features = text_features.cpu().detach().numpy()

            for im_ft, tx_ft, id_ft, caption_ft in zip(image_features, text_features, ids, captions):
                train_img_features.append(im_ft)
                train_text_features.append(tx_ft)
                labels_train.append(id_ft)
                train_captions.append(caption_ft)

        with open('train_image_emb' + prefix + '.pkl', 'wb') as f:
            pickle.dump(train_img_features, f)
        
        with open('train_text_emb' + prefix + '.pkl', 'wb') as f:
            pickle.dump(train_text_features, f)

        with open('train_labels' + prefix + '.pkl', 'wb') as f:
            pickle.dump(labels_train, f)
        
        with open('train_captions' + prefix + '.pkl', 'wb') as f:
            pickle.dump(train_captions, f)
        '''

        del train_img_features
        del train_text_features
        del labels_train
        del train_captions
        del train_dataloader
        del train_data

        # ---------------------------------------------------------------

        with open('val_dataset_COCO.pkl', 'rb') as f:
            val_data = pickle.load(f)

        val_data.text_model = modelText
    
        val_dataloader = DataLoader(dataset=val_data, 
                                batch_size=batch_size, 
                                shuffle=False) # don't usually need to shuffle testing data

        labels_val = []
        val_img_features = []
        val_text_features = []
        val_captions = []

        for images, captions, text_features, ids in tqdm(val_dataloader):
            images = images.to(device)
            text_features = torch.tensor(text_features).to(device)

            image_features = model_image(images)
            text_features = model_text(text_features)

            image_features = image_features.cpu().detach().numpy()
            text_features = text_features.cpu().detach().numpy()

            for im_ft, tx_ft, id_ft, caption_ft in zip(image_features, text_features, ids, captions):
                val_img_features.append(im_ft)
                val_text_features.append(tx_ft)
                labels_val.append(id_ft)
                val_captions.append(caption_ft)

        with open('val_image_emb' + prefix + '.pkl', 'wb') as f:
            pickle.dump(val_img_features, f)

        with open('val_text_emb' + prefix + '.pkl', 'wb') as f:
            pickle.dump(val_text_features, f)

        with open('val_labels' + prefix + '.pkl', 'wb') as f:
            pickle.dump(labels_val, f)
        
        with open('val_captions' + prefix + '.pkl', 'wb') as f:
            pickle.dump(val_captions, f)

        # ---------------------------------------------------------------

def get_unique(images, labels):
    unique_pairs = {}

    # Iterate through each label and image pair
    for label, image in zip(labels, images):
        # If the label is not already in the dictionary, add it with its corresponding image
        if label.item() not in unique_pairs:
            unique_pairs[label.item()] = image

    # Extract the unique labels and images from the dictionary
    unique_labels = list(unique_pairs.keys())
    unique_images = list(unique_pairs.values())

    return unique_images, unique_labels

if __name__ == '__main__':
    image_weights = 'weights/model_image_text2img_bueno.pth'
    text_weights = 'weights/model_text_text2img_bueno.pth'
    prefix = 'txt2img'

    generate_data(image_weights, text_weights, prefix)
