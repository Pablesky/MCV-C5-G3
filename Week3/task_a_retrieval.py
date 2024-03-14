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

from metrics import precission_k, recall_k, ap

class Retrieval:
    def __init__(self, retrieval_method, retrieval_parameters, train_features, train_labels, test_features, test_labels, train_paths, test_paths):

        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.train_paths = train_paths
        self.test_paths = test_paths

        self.counterGT = None

        self.retrieval_method = retrieval_method
        
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        if retrieval_method == 'KNN':
            self.model = KNeighborsClassifier(**retrieval_parameters)

        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        elif retrieval_method == 'KMEANS':
            self.model = KMeans(**retrieval_parameters)
        
        # https://github.com/facebookresearch/faiss
        elif retrieval_method == 'FAISS':
            pass
    
    def fit(self):
        if self.retrieval_method == 'KNN':
            self.model.fit(self.train_features, self.train_labels)

        elif self.retrieval_method == 'KMEANS':
            self.transformed_train_feats = self.model.fit_transform(self.train_features, self.train_labels)
        
        elif self.retrieval_method == 'FAISS':
            pass

    def plot_top_5(self, dist, actual_image, folder):
        sorted_indices = np.argsort(dist)

        top_paths = [self.train_paths[idx] for idx in sorted_indices[:5]]
        top_images = [Image.open(path) for path in top_paths]

        input_image = Image.open(self.test_paths[actual_image])

        reset_folder(folder)

        plot_images(input_image, top_images, actual_image, folder)

    def compute_metrics(self, top_labels, actual_label):
        # Comun para todas las evaluaciones
        prec_1 = precission_k(actual_label, [top_labels[0]])
        prec_5 = precission_k(actual_label, top_labels)

        if self.counterGT is None:
            unique, counts = np.unique(self.train_labels, return_counts=True)
            self.counterGT = dict(zip(unique, counts))


        rec_1 = recall_k(actual_label, [top_labels[0]], self.counterGT[actual_label])
        rec_5 = recall_k(actual_label, top_labels, self.counterGT[actual_label])

        print(prec_1, prec_5)
        print(rec_1, rec_5)

        return prec_1

    def retrieve_and_eval(self):
        if self.retrieval_method == "KNN":
            # quin tipus de distancia??

            # self.dists2query -> distance
            # self.indexes_of_dists2query -> index of the train_features

            self.dists2query, self.indexes_of_dists2query = self.model.kneighbors(X=self.test_features, n_neighbors=5, return_distance=True)

            # reset_folder('KNN')

            # Already sorted
            for i, (dist, index) in enumerate(zip(self.dists2query, self.indexes_of_dists2query)):
                
                top_labels = [self.train_labels[idx] for idx in index[:5]]
                actual_label = self.test_labels[i]

                print(top_labels)
                print(actual_label)

                metrics = self.compute_metrics(top_labels, actual_label)

                # To save the predictions
                # plot_images(Image.open(self.test_paths[i]), [Image.open(self.train_paths[i]) for i in index], i, 'KNN')


        if self.retrieval_method == "KMEANS":
            self.transformed_test_feats = self.model.transform(X=self.test_features)

            # reset_folder('KMEANS')

            for i, test_feat in enumerate(self.transformed_test_feats):
                dist = np.linalg.norm(self.transformed_train_feats - test_feat, axis=1)
                sorted_indices = np.argsort(dist)

                top_labels = [self.train_labels[idx] for idx in sorted_indices[:5]]
                actual_label = self.test_labels[i]

                metrics = self.compute_metrics(top_labels, actual_label)

                # To save the predictions
                # self.plot_top_5(dist, i, 'KMEANS')

        if self.retrieval_method == "FAISS":
            pass



# Format of the pickle file
# First dimension: 0-> test, 1-> train
# Second dimension: 0-> feature, 1-> label
# Third dimension: N_sample

    
if __name__ == '__main__':

    with open('paths.pkl', 'rb') as f:
        path_images = pickle.load(f)

    train_paths = path_images[1]
    test_paths = path_images[0]
    
    with open('datasetResnet152.pkl', 'rb') as f:
        dataset_features = pickle.load(f)

    train_features = dataset_features[1][0]
    train_labels = dataset_features[1][1]

    test_features = dataset_features[0][0]
    test_labels = dataset_features[0][1]

    
    params = {

    }

    retr = Retrieval(retrieval_method = 'KNN', 
                     retrieval_parameters = params, 
                     train_features = train_features, 
                     train_labels = train_labels, 
                     test_features = test_features, 
                     test_labels = test_labels,
                     train_paths=train_paths,
                     test_paths=test_paths)
    
    retr.fit()

    retr.retrieve_and_eval()