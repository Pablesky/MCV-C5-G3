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
from sklearn.manifold import TSNE
import seaborn as sns
from torchvision.models import resnet152, ResNet152_Weights

import os
import shutil

from utils_w4 import *

import torch

import faiss

import time

from sklearn.metrics import confusion_matrix

from utils_w4 import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from sklearn.metrics import average_precision_score


def compute_metrics(true_class, class_preds, number_true):

    positive_accumulation = 0
    ap = 0
    rec = []
    prec = []

    for i in range(len(class_preds)):

        if true_class == class_preds[i]:
            positive_accumulation += 1
            ap += positive_accumulation / (i + 1)
        
        prec.append(positive_accumulation / (i + 1))
        rec.append(positive_accumulation / number_true)
    
    ap = ap / number_true

    return ap, prec, rec

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        
        weights = ResNet152_Weights.DEFAULT
        self.model = resnet152(weights=weights).to(device)
        self.model.fc = nn.Identity()

    def forward(self, x):
        output = self.model(x)
        return output

class Visualize:
    def __init__(self, train_features, train_labels, test_features, test_labels, vis_type, n_comp):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.vis_type = vis_type
        self.n_comp = n_comp
        """if self.vis_type == "TSNE":
            self.visualizer = TSNE(n_components=self.n_comp, learning_rate='auto', init='random', perplexity=3).fit(self.train_features)"""
    
    def visualization(self):

        if self.vis_type == "PCA":
            pca = PCA(n_components=2).fit(self.train_features)
            train_embedded = pca.transform(self.train_features)
            test_embedded = pca.transform(self.test_features)


        if self.vis_type == "TSNE":
            train_embedded = TSNE(n_components=self.n_comp, learning_rate='auto', init='random', perplexity=3).fit_transform(self.train_features)

        if self.vis_type == 'UMAP':
            reducer = umap.UMAP(n_components=self.n_comp)

            train_scaled_data = StandardScaler().fit_transform(self.train_features)
            test_scaled_data = StandardScaler().fit_transform(self.test_features)
            
            train_embedded = reducer.fit_transform(train_scaled_data)
            test_embedded = reducer.fit_transform(test_scaled_data)
            
        
        if self.n_comp == 2:
            sns.scatterplot(x=train_embedded[:, 0], y=train_embedded[:, 1], hue=self.train_labels, palette='tab10', legend=False)
            plt.xlabel(f'{self.vis_type} Component 1')
            plt.ylabel(f'{self.vis_type} Component 2')
            plt.title('TEST')
            plt.savefig(f'image1.png')
            plt.close()

        if self.n_comp >= 3:

            class_colors = {
                0 : 'blue',
                1: 'red',
                2: 'green',
                3: 'orange',
                4: 'purple',
                5: 'yellow',
                6: 'cyan',
                7: 'magenta',
            }

            classes_names = ['coast', 'forest', 'highway', 'insidecity', 'mountain', 'Opencountry', 'street', 'tallbuilding']

            colors = [class_colors[c] for c in self.train_labels]

            
            fig1 = go.Figure(data=[go.Scatter3d(x=train_embedded[:, 0], y=train_embedded[:, 1], z=train_embedded[:, 2],
                                   mode='markers',
                                   marker=dict(color=colors, size=3)
                                   )])
            
            fig1.update_layout(title=f'3D Train {self.vis_type} Visualization')
            fig1.update_layout(showlegend=True)
            fig1.savefig(f'image1.png')
            fig1.close()

            colors = [class_colors[c] for c in self.test_labels]

            
            fig2 = go.Figure(data=[go.Scatter3d(x=test_embedded[:, 0], y=test_embedded[:, 1], z=test_embedded[:, 2],
                                   mode='markers',
                                   marker=dict(color=colors, size=3)
                                   )])
            
            fig2.update_layout(title=f'3D Test {self.vis_type} Visualization')
            fig2.update_layout(showlegend=True)
            fig2.savefig(f'image2.png')
            fig2.close()

class Retrieval:
    def __init__(self, image_emb, text_emb, labels, captions, plot, case, k):

        """self.train_features = train_features
        self.test_features = test_features
        """

        self.image_emb = image_emb
        self.text_emb = text_emb
        self.labels = labels
        self.captions = captions

        self.image_emb = np.array(self.image_emb)
        self.text_emb = np.array(self.text_emb)
        self.labels = np.array(self.labels)
        self.captions = np.array(self.captions)

        self.retrieval_method = "FAISS"

        self.num_elems = k

        self.save_predictions = {}

        self.plot = plot
        self.case = case

        self.path = self.retrieval_method + self.case + 'image2text'

        self.model_image = EmbeddingNet()
        self.model_image.load_state_dict(torch.load('./model_image_bert_a.pth'))
        self.model_image = self.model_image.to(device)
        self.model_image.eval()

        weights = ResNet152_Weights.DEFAULT
        self.transform = weights.IMAGENET1K_V2.transforms()

        # https://github.com/facebookresearch/faiss

        self.model = faiss.IndexFlatL2(len(self.image_emb[0]))

    
    def fit(self):
        self.model.add(self.image_emb)


    def compute_metrics(self, predicted_labels, index_predictions, actual_label, actual_index, plot = False):
        if plot: 
            query_image = Image.open(f'/ghome/mcv/datasets/C5/COCO/{self.case}2014/COCO_{self.case}2014_{str(actual_label).zfill(12)}.jpg')
            plt.imshow(query_image)
            similar_text = [self.captions[i] for i in index_predictions]
        
            plot_texts(query_image = query_image, 
                    query_index = actual_index, 
                    captions = similar_text,
                    real_caption = self.captions[actual_index],
                    path = self.path)

        
        # Comun para todas las evaluaciones
            
        if actual_label not in self.save_predictions:
            self.save_predictions[actual_label] = []

        self.save_predictions[actual_label].append(predicted_labels)
        
             

    def retrieve_and_eval(self):
        if self.plot:
            reset_folder(self.path)


        for i, test_feature in enumerate(self.image_emb):

            actual_feature = test_feature.reshape(1, -1)

            _, indices = self.model.search(actual_feature, 30)

            indices_subset = indices[0][5::5]

            top_labels = [self.labels[idx] for idx in indices_subset]
            actual_label = self.labels[i]

            self.compute_metrics(
                predicted_labels = top_labels, 
                index_predictions = indices_subset, 
                actual_label = actual_label, 
                actual_index = i, 
                plot = self.plot
            )

            if i == self.num_elems-1:
                break
                        
        
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

    def custom_image(self, image_path):
        image = read_image(image_path, ImageReadMode.RGB)
        image = self.transform(image).unsqueeze(0).to(device)
        image_emb = self.model_image(image).detach().cpu().numpy()

        _, indices = self.model.search(image_emb, 25)

        indices_subset = indices[0][0::5]

        similar_text = [self.captions[i] for i in indices_subset]

        plot_texts_custom(query_image = Image.open(image_path), 
                    name = image_path.split('/')[-1], 
                    captions = similar_text,
                    path = './images_custom2')


        
if __name__ == '__main__':
    case = 'val'
    prefix = 'big_img2txt' 
    k = 300
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

    else:
        with open('val_image_emb' + prefix + '.pkl', 'rb') as f:
            image_emb = pickle.load(f)

        with open('val_text_emb' + prefix + '.pkl', 'rb') as f:
            text_emb = pickle.load(f)

        with open('val_labels' + prefix + '.pkl', 'rb') as f:
            labels = pickle.load(f)

        with open('val_captions' + prefix + '.pkl', 'rb') as f:
            captions = pickle.load(f)

    if k > 0:
        retr = Retrieval(
                        image_emb = image_emb,
                        text_emb = text_emb,
                        labels = labels,
                        captions = captions,
                        plot = plot,
                        case = case,
                        k = k)
    else:
        retr = Retrieval(
                    image_emb = image_emb,
                    text_emb = text_emb,
                    labels = labels,
                    captions = captions,
                    plot = plot,
                    case = case,
                    k = k)
    
    retr.fit()
    retr.custom_image('images_custom/4.jpeg')
    #retr.retrieve_and_eval()

    #patata = Visualize(train_features = np.array(text_emb[:1000]), train_labels = labels[:1000], test_features = np.array(text_emb), test_labels = labels, vis_type = 'TSNE',n_comp = 2)
    #patata.visualization()