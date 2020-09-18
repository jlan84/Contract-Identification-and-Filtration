from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from pipeline import ContractPipeline
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class KMeansCluster():

    def __init__(self, documents, target_lst, max_features=None, n_clusters=8):
        """
        Instantiates a KMeansCluster object
        
        Params
        documents: list of strings to be analyzed
        max_features: int max number of features
        n_clusters: int number of clusters to uses
        """
        self.max_features = max_features
        self.n_clusters = n_clusters
        self.documents = documents
        self.target_lst = target_lst


    def get_kmeans(self):
        """
        Generates a document term matrix (X) mapped array with feature indeces
        mapped to feature names (features) kmeans obejct (kmeans) and fits X to
        the kmeans object
        """
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        self.X = self.vectorizer.fit_transform(self.documents)
        self.features = self.vectorizer.get_feature_names()
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.kmeans.fit(self.X)
    

    def get_silhouette_score(self, max_features=None, n_clusters=8):
        """
        Calculates the silhouette average score (silhouette_avg) for the kmeans model and computes cluster centers 
        and indexes (cluster_labels)
        Params
        max_features: int max number of features to consider for the tfidf
        n_clusters: int number of clusters for the kmeans model
        """
        vectorizer = TfidfVectorizer(max_features=max_features)
        X = vectorizer.fit_transform(self.documents)
        features = vectorizer.get_feature_names()
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(f'The silhouette score for {n_clusters} clusters is {silhouette_avg}')
        return silhouette_avg
        

    def print_top_features_kmeans(self):
        """
        Prints and adds to dictionary (top_feature_dic) the top features for each cluster
        """
        top_centroids = self.kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
        print('Top words for each cluster:')
        self.top_feature_dic = {}
        for num, centroid in enumerate(top_centroids):
            self.top_feature_dic[num] = [self.features[i] for i in centroid]
            print(f'{num}, {", ".join(self.features[i] for i in centroid)}')

    def print_most_common_contracts_in_cluster(self):
        """
        Prints out and adds to cluster_dic the targest associated with each cluster
        """
        assigned_cluster = self.kmeans.transform(self.X).argmin(axis=1)
        self.cluster_dic = {}
        for i in range(self.kmeans.n_clusters):
            cluster = np.arange(0, self.X.shape[0])[assigned_cluster==i]
            target = [self.target_lst[j] for j in cluster]
            most_common = Counter(target).most_common()
            self.cluster_dic[f'Cluster {i}'] = most_common
            print(f'Cluster {i}')
            for k in range(len(most_common)):
                print(f"     {most_common[k][0]} ({most_common[k][1]} contracts)")

    def plot_silhouettes(self, start, stop, step):
        """
        Generates a line plot for the cluster scores from (start) to (stop) stepping by (step) 
        amount

        Params
        start: int start of the clusters for the kmeans model (must be > 1)
        stop: int stop number of clusters
        step: int step size for for the clusters
        """
        sil_avg_lst = []
        for i in range(start,stop,step):
            avg = self.get_silhouette_score(n_clusters=i)
            sil_avg_lst.append(avg)

        fig, ax = plt.subplots(figsize=(12,10))

        x = np.arange(start,stop,step)
        ax.plot(x, sil_avg_lst)
        ax.set_title('Silhouette Score', fontsize=20)
        ax.set_xlabel('Number of clusters', fontsize=16)
        ax.set_ylabel('Score', fontsize=16)
        plt.show()

if __name__ == "__main__":
    directory = '/Users/justinlansdale/Documents/Galvanize/Capstone2/\
Contract-Classifier/data/CurrentContracts/'

    stop_words = stopwords.words('english')
    # stop_words.append('name')
    # stop_words.append('date')
    # stop_words.append('buyer')
    # stop_words.append('landlord')
    # stop_words.append('seller')
    pipe = ContractPipeline(directory, stop_words)
    
    pipe.get_list_of_docs()
    pipe.remove_stop_words()
    pipe.word_condenser()
    count_matrix, tfidf_matrix, cv = pipe.tf_idf_matrix(pipe.porter_str)

    kmc = KMeansCluster(pipe.porter_str, pipe.target_lst)
    kmc.get_kmeans()
    kmc.get_silhouette_score()
    kmc.print_top_features_kmeans()
    kmc.print_most_common_contracts_in_cluster()
    kmc.plot_silhouettes(2,26,3)
    
    top_features_df = pd.DataFrame(kmc.top_feature_dic)
    print(top_features_df.to_markdown())
