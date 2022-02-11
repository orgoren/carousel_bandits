from numpy import array, shape
import pandas as pd
import numpy as np
import argparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import math


# Defining a class for clustering playlists
class PlaylistClusterer:

    # Initializing
    def __init__(self, features_file):
        #open the playlists and load as arrays
        print(f"reading {features_file} file")
        features_df = pd.read_csv(features_file)
        print("done")
        self.centroids = None
        self.labels = None
        self.k = -1
        self.features = np.array(features_df)


    def get_sse_score(self):
        assert self.k != -1
        sse = 0
        for i in range(len(self.features)):
            sse += np.linalg.norm(self.features[i]-self.centroids[self.labels[i]], ord=2)

        return sse


    def cluster_2d(self, k=10):
        # Load Data
        data = self.features
        pca = PCA(2)

        # Transform the data
        df = pca.fit_transform(data)

        # Initialize the class object
        kmeans = KMeans(n_clusters=k)

        # predict the labels of clusters.
        label = kmeans.fit_predict(df)

        self.centroids = kmeans.cluster_centers_

        # Getting unique labels
        u_labels = np.unique(label)

        # plotting the results:
        for i in u_labels:
            plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
        plt.legend()
        plt.show()

    def split_small_clusters(self, small_threshold=3):
        data = self.features
        new_label_name = self.k
        new_labels = self.labels.copy()

        small_cluster_labels = set()
        new_small_clusters_number = 0

        for i in self.u_labels:
            cluster_size = len(data[self.labels == i])
            print(f"For cluster {i}, size {cluster_size}")
            if cluster_size < small_threshold:
                new_small_clusters_number += cluster_size + 1
                small_cluster_labels.add(i)


        new_big_clusters_number = self.k - len(small_cluster_labels)
        print(f"Got {len(small_cluster_labels)} small clusters and {new_big_clusters_number} big clusters")

        for ind,val in enumerate(self.labels):
            if val in small_cluster_labels:
                new_labels[ind] = new_label_name
                self.centroids = np.concatenate((self.centroids, self.features[ind].reshape(1,97)),axis=0)
                new_label_name += 1
                assert self.centroids.shape[0] == new_label_name

        if self.centroids.shape[0] != new_small_clusters_number + new_big_clusters_number:
            print(f"c={self.centroids.shape[0]} ns={new_small_clusters_number} nb={new_big_clusters_number}")
            exit(1)

        if self.k > 100:
            print("good")

        self.labels = new_labels
        return new_labels

    def cluster(self, k=10, split_small_clusters=False):
        self.k = k

        # Initialize the class object
        kmeans = KMeans(n_clusters=self.k)

        # predict the labels of clusters.
        self.labels = kmeans.fit_predict(self.features)
        self.centroids = kmeans.cluster_centers_

        # Getting unique labels
        self.u_labels = np.unique(self.labels)

        if split_small_clusters:
            return self.split_small_clusters()
        else:
            for i in self.u_labels:
               print(f"For cluster {i}, size {len(self.features[self.labels == i])}")

        return self.labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--playlists_path", type=str, default="data/playlist_features.csv", required=False,
                        help="Path to playlist features file")
    parser.add_argument("--num_clusters", type=int, default=20, required=False, help="Number of clusters")

    args = parser.parse_args()

    clusterer = PlaylistClusterer(args.playlists_path)

    scores = []

    for i in range(20,400,3):
        print(f"clustering {i}")
        label = clusterer.cluster(i)
        #print(f"Score for {i} is {score}")
        scores.append(i)

    plt.plot(scores)
    plt.show()
    print("hi")



    #clusterer.cluster(args.num_clusters)
