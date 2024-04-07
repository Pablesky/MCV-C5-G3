import pickle
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import average_precision_score

from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
from PIL import Image
from torch import nn
import numpy as np
from torchvision.io import ImageReadMode
from torchvision.io import read_image

import matplotlib.pyplot as plt

from torchvision.models import resnet152, ResNet152_Weights

import os
import shutil

from utils import reset_folder

import torch

from metrics import compute_metrics

import faiss

import time

from sklearn.metrics import confusion_matrix

from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

class Retrieval:
    def __init__(self, image_emb, text_emb, labels, captions, plot, case, unique_labels):

        """self.train_features = train_features
        self.test_features = test_features
        """
        self.unique_labels = unique_labels
        self.image_emb = image_emb
        self.text_emb = text_emb
        self.labels = labels
        self.captions = captions

        self.image_emb = np.array(self.image_emb)
        self.text_emb = np.array(self.text_emb)
        self.labels = np.array(self.labels)
        self.captions = np.array(self.captions)
        self.unique_labels = np.array(self.unique_labels)

        self.retrieval_method = "FAISS"

        self.num_elems = len(self.image_emb)

        self.save_predictions = {}

        self.plot = plot
        self.case = case

        self.path = self.retrieval_method + self.case + 'text2img'

        self.model_text = Fast2Res()
        self.model_text.load_state_dict(torch.load('weights/model_text_text2img_bueno.pth'))
        self.model_text = self.model_text.to(device)
        self.model_text.eval()

        self.fastText = fasttext.load_model("fasttext_wiki.en.bin")

        # https://github.com/facebookresearch/faiss

        self.model = faiss.IndexFlatL2(len(self.image_emb[0]))

    
    def fit(self):
        self.model.add(self.image_emb)


    def compute_metrics(self, predicted_labels, index_predictions, actual_label, actual_index, plot = False):
        if plot: 
            query_text = self.captions[actual_index]
            similar_images = [Image.open(f'../COCO/{self.case}2014/COCO_{self.case}2014_{str(i).zfill(12)}.jpg') for i in predicted_labels[:5]]
        
            plot_image(query_text = query_text, 
                    query_index = actual_index, 
                    images = similar_images,
                    real_image = Image.open(f'../COCO/{self.case}2014/COCO_{self.case}2014_{str(actual_label).zfill(12)}.jpg'),
                    path = self.path)

        
        # Comun para todas las evaluaciones
            
        if actual_label not in self.save_predictions:
            self.save_predictions[actual_label] = []

        self.save_predictions[actual_label].append(predicted_labels)
        
             

    def retrieve_and_eval(self):
        if self.plot:
            reset_folder(self.path)


        for i, test_feature in enumerate(self.text_emb):

            actual_feature = test_feature.reshape(1, -1)

            _, indices = self.model.search(actual_feature, self.num_elems)

            top_labels = [self.unique_labels[idx] for idx in indices[0]]
            print(top_labels[:10])
            actual_label = self.labels[i]

            self.compute_metrics(
                predicted_labels = top_labels, 
                index_predictions = indices[0], 
                actual_label = actual_label, 
                actual_index = i, 
                plot = self.plot
            )
        
        # Seguramente algo aqui hay que hacerlo comun
        precissions_per_class = []
        recalls_per_class = []
        ap_per_class = []

        for actual_label in self.save_predictions:
            predictions = np.array(self.save_predictions[actual_label])

            num_positives = np.sum(self.labels == actual_label)

            precisions_list = []
            recall_list = []
            ap_list = []

            for prediction in predictions:

                ap, prec, rec = compute_metrics(actual_label, prediction, num_positives)

                precisions_list.append(prec)
                recall_list.append(rec)
                ap_list.append(ap)

            precissions_per_class.append(precisions_list)
            recalls_per_class.append(recall_list)
            ap_per_class.append(ap_list)

        prec_at_k = []
        recall_at_k = []
        ap_at_k = []
        
        for i in range(len(precissions_per_class)):

            precissions_mean = np.mean(precissions_per_class[i], axis=0)
            prec_at_k.append(precissions_mean)

            recalls_mean = np.mean(recalls_per_class[i], axis=0)
            recall_at_k.append(recalls_mean)

            ap_mean = np.mean(ap_per_class[i])
            ap_at_k.append(ap_mean)

        mean_prec_at_k = np.mean(prec_at_k, axis=0)
        mean_recall_at_k = np.mean(recall_at_k, axis=0)
        ap_at_k = np.mean(ap_at_k)

        plt.cla()

        plt.plot(mean_recall_at_k, mean_prec_at_k)

        plt.title('Precission-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precission')
        plt.savefig(f'precission_recall{self.path}.png')
        plt.close()

        print('Precission at 1: ', mean_prec_at_k[0])
        print('Precission at 5: ', mean_recall_at_k[4])
        print('mAP: ', ap_at_k)

    def custom_text(self, text):
        cleaned_sentence = clean_sentence(self.fastText, text)
        text_emb = self.fastText.get_sentence_vector(cleaned_sentence)
        text_emb = self.model_text(torch.tensor(text_emb).unsqueeze(0).to(device)).detach().cpu().numpy()

        _, indices = self.model.search(text_emb, 5)

        similar_images = [Image.open(f'../COCO/{self.case}2014/COCO_{self.case}2014_{str(self.unique_labels[i]).zfill(12)}.jpg') for i in indices[0]]

        plot_image_custom(query_text = cleaned_sentence, 
                    name = clean_sentence, 
                    images = similar_images,
                    path = 'customTexts_results')


        
if __name__ == '__main__':
    case = 'val'
    prefix = 'img2txt' # txt2img
    k = 10
    plot = True
    
    if case == 'train':

        with open('train_image_emb' + prefix + '.pkl', 'rb') as f:
            image_emb = pickle.load(f)

        with open('train_text_emb' + prefix + '.pkl', 'rb') as f:
            text_emb = pickle.load(f)

        with open('train_labels' + prefix + '.pkl', 'rb') as f:
            labels = pickle.load(f)

        with open('train_captions' + prefix + '.pkl', 'rb') as f:
            captions = pickle.load(f)
        
        unique_images, unique_labels = get_unique(image_emb, labels)

    else:

        with open('val_image_emb' + prefix + '.pkl', 'rb') as f:
            image_emb = pickle.load(f)

        with open('val_text_emb' + prefix + '.pkl', 'rb') as f:
            text_emb = pickle.load(f)

        with open('val_labels' + prefix + '.pkl', 'rb') as f:
            labels = pickle.load(f)

        with open('val_captions' + prefix + '.pkl', 'rb') as f:
            captions = pickle.load(f)

        unique_images, unique_labels = get_unique(image_emb, labels)

    if k > 0:
        retr = Retrieval(
                        image_emb = unique_images,
                        text_emb = text_emb[:k],
                        labels = labels[:k],
                        captions = captions[:k],
                        plot = plot,
                        case = case,
                        unique_labels = unique_labels)
    else:
        retr = Retrieval(
                    image_emb = unique_images,
                    text_emb = text_emb,
                    labels = labels,
                    captions = captions,
                    plot = plot,
                    case = case,
                    unique_labels = unique_labels)
    
    retr.fit()
    retr.retrieve_and_eval()

    '''
    while True:
        text = input('Introduce una frase: ')
        if text == 'exit':
            break
        retr.custom_text(text)
    '''
    
    