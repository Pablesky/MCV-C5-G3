import pickle
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import average_precision_score

from sklearn.metrics import classification_report, accuracy_score

# https://github.com/facebookresearch/faiss


class Retrieval:
    def __init__(self, retrieval_method, retrieval_parameters, train_features, train_labels, test_features, test_labels):

        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels

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
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        if self.retrieval_method == 'KNN':
            self.model.fit(self.train_features, self.train_labels)

        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        elif self.retrieval_method == 'KMEANS':
            self.transformed_train_feats = self.model.fit_transform(self.train_features, self.train_labels)
        
        # https://github.com/facebookresearch/faiss
        elif self.retrieval_method == 'FAISS':
            pass

    def retrieve(self):
        if self.retrieval_method == "KNN":
            # quin tipus de distancia??
            self.dists2query, self.indexes_of_dists2query = self.model.kneighbors([self.test_features, 5, True])

        if self.retrieval_method == "KMEANS":
            self.transformed_test_feats = self.model.transform(self.test_features)

            #compute distances
            
            
        if self.retrieval_method == "FAISS":
            pass
    
        

    def eval(self):
        #MAP, prec@1, prec@5
        pass


# Format of the pickle file
# First dimension: 0-> test, 1-> train
# Second dimension: 0-> feature, 1-> label
# Third dimension: N_sample
    
if __name__ == '__main__':

    with open('datasetResnet152.pkl', 'rb') as f:
        dataset_features = pickle.load(f)

    train_features = dataset_features[1][0]
    train_labels = dataset_features[1][1]

    test_features = dataset_features[0][0]
    test_labels = dataset_features[0][1]

    retr = Retrieval(retrieval_method='FAISS')

    