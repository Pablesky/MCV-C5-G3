import pickle
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import average_precision_score

from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

import os
import shutil

from utils import plot_images, reset_folder

from metrics import compute_metrics

import faiss



class Retrieval:
    def __init__(self, retrieval_method, retrieval_parameters, train_features, train_labels, test_features, test_labels, train_paths, test_paths, plot):

        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.train_paths = train_paths
        self.test_paths = test_paths

        self.num_train = self.train_features.shape[0]
        self.num_test = self.test_features.shape[0]

        self.retrieval_method = retrieval_method

        self.save_predictions = {}

        self.plot = plot
        
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        if retrieval_method == 'KNN':
            self.model = KNeighborsClassifier(**retrieval_parameters)

        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        elif retrieval_method == 'KMEANS':
            self.model = KMeans(**retrieval_parameters)
        
        # https://github.com/facebookresearch/faiss
        elif retrieval_method == 'FAISS':
            self.model = faiss.IndexFlatL2(len(self.train_features[0]))
    
    def fit(self):
        if self.retrieval_method == 'KNN':
            self.model.fit(self.train_features, self.train_labels)

        elif self.retrieval_method == 'KMEANS':
            self.transformed_train_feats = self.model.fit_transform(self.train_features, self.train_labels)
        
        elif self.retrieval_method == 'FAISS':
            self.model.add(self.train_features)

    def compute_metrics(self, predicted_labels, index_predictions, actual_label, actual_index, plot = False):
        if plot:
            
            plot_images(
                query_image = Image.open(self.test_paths[actual_index].replace('\\', '/')), 
                query_index = actual_index, 
                query_label = actual_label,
                similar_images = [Image.open(self.train_paths[i].replace('\\', '/')) for i in index_predictions[:5]], 
                images_labels = predicted_labels[:5], 
                path = self.retrieval_method
            )
        
        # Comun para todas las evaluaciones
            
        if actual_label not in self.save_predictions:
            self.save_predictions[actual_label] = []

        self.save_predictions[actual_label].append(predicted_labels)
        
             

    def retrieve_and_eval(self):

        if self.plot:
            reset_folder(self.retrieval_method)

        if self.retrieval_method == "KNN":

            self.dists2query, self.indexes_of_dists2query = self.model.kneighbors(X=self.test_features, n_neighbors=self.num_train, return_distance=True)

            for i, (dist, index) in enumerate(zip(self.dists2query, self.indexes_of_dists2query)):
                
                predicted_labels = [self.train_labels[idx] for idx in index]
                actual_label = self.test_labels[i]

                self.compute_metrics(
                    predicted_labels = predicted_labels, 
                    index_predictions = index, 
                    actual_label = self.test_labels[i], 
                    actual_index = i, 
                    plot = self.plot
                )

        elif self.retrieval_method == "KMEANS":
            self.transformed_test_feats = self.model.transform(X=self.test_features)

            for i, test_feat in enumerate(self.transformed_test_feats):
                dist = np.linalg.norm(self.transformed_train_feats - test_feat, axis=1)
                sorted_indices = np.argsort(dist)

                predicted_labels = [self.train_labels[idx] for idx in sorted_indices]

                self.compute_metrics(
                    predicted_labels = predicted_labels, 
                    index_predictions = sorted_indices, 
                    actual_label = self.test_labels[i], 
                    actual_index = i, 
                    plot = self.plot
                )

        elif self.retrieval_method == "FAISS":

            for i, test_feature in enumerate(self.test_features):

                actual_feature = test_feature.reshape(1, -1)
                k = self.num_train 
                _, indices = self.model.search(actual_feature, k)

                top_labels = [self.train_labels[idx] for idx in indices[0]]
                actual_label = self.test_labels[i]

                self.compute_metrics(
                    predicted_labels = top_labels, 
                    index_predictions = indices[0], 
                    actual_label = actual_label, 
                    actual_index = i, 
                    plot = self.plot
                )

        else:
            raise ValueError("Not this type of retrieval method is implemented yet.")
        
        # Seguramente algo aqui hay que hacerlo comun
        precissions_per_class = []
        recalls_per_class = []
        ap_per_class = []

        for actual_label in self.save_predictions:
            predictions = np.array(self.save_predictions[actual_label])

            num_positives = np.sum(self.train_labels == actual_label)

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

            plt.plot(recalls_mean, precissions_mean)

        plt.title('Precission-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precission')
        plt.legend(['coast', 'forest', 'highway', 'insidecity', 'mountain', 'opencountry', 'street', 'tallbuilding'])
        plt.savefig(f'precission_recall{self.retrieval_method}.png')

        class_list = ['coast', 'forest', 'highway', 'insidecity', 'mountain', 'opencountry', 'street', 'tallbuilding']

        for i, class_id in enumerate(class_list):
            print(f'{class_id} Precission at 1: ', prec_at_k[i][0])
            print(f'{class_id} Precission at 5: ', prec_at_k[i][4])
            print(f'{class_id} mAP: ', ap_at_k[i])

        map_score = np.mean(ap_at_k)

        print('Mean average precission: ', map_score)

        
if __name__ == '__main__':

    with open('paths.pkl', 'rb') as f:
        path_images = pickle.load(f)

    train_paths = path_images[1]
    test_paths = path_images[0]
    
    with open('siamesePredictions.pkl', 'rb') as f:
        dataset_features = pickle.load(f)

    train_features = np.array(dataset_features[1][0])
    train_labels = np.array(dataset_features[1][1])

    test_features = np.array(dataset_features[0][0])
    test_labels = np.array(dataset_features[0][1])
    
    params = {

    }

    retr = Retrieval(retrieval_method = 'FAISS', 
                     retrieval_parameters = params, 
                     train_features = train_features, 
                     train_labels = train_labels, 
                     test_features = test_features, 
                     test_labels = test_labels,
                     train_paths=train_paths,
                     test_paths=test_paths,
                     plot=True)
    
    retr.fit()

    retr.retrieve_and_eval()