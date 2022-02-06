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
        features_df = pd.read_csv(features_file)
        self.features = np.array(features_df)


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

        # Getting unique labels
        u_labels = np.unique(label)

        # plotting the results:
        for i in u_labels:
            plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
        plt.legend()
        plt.show()

    def cluster(self, k=10, debug=False):

        # Load Data
        data = self.features


        # Initialize the class object
        kmeans = KMeans(n_clusters=k)

        # predict the labels of clusters.
        label = kmeans.fit_predict(data)

        # Getting unique labels
        u_labels = np.unique(label)

        if debug:
            # number of results per group:
            for i in u_labels:
                print(f"For cluster {i}, size {len(data[label == i])}")
        return label


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
