import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import json
from cluster_playlists import PlaylistClusterer

def calculate_WSS(kmeans_model : PlaylistClusterer, kmax, split_small_clusters, initial=10, interval=10):
    wss = []
    for k in range(initial, kmax+1, interval):
        labels, centroids = kmeans_model.cluster(k, split_small_clusters)
        print(f"working on k={k} #centriods={centroids.shape[0]}")
        wss.append(kmeans_model.get_sse_score(centroids, labels))
    return wss


def get_cluster_sse_results(kmeans_model, res_file, kmax=400, split_small_clusters=False, initial=10, interval=10):
    wss = calculate_WSS(kmeans_model, kmax, split_small_clusters=split_small_clusters, initial=initial, interval=interval)
    with open(res_file, "w") as f:
        json.dump(wss, f)

def plot_graph(res, title, xlabel, ylabel):
    print(title)
    for i, s in enumerate(res):
        print(f"{i} : {s}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot([i*10 for i in range(len(res))], res)
    plt.savefig(f"{title}.jpg")

def show_results(res_file):
    with open(res_file) as f:
        res = json.load(f)

    incs = [res[i+1]-res[i] for i in range(len(res[:-1]))]

    plot_graph(res, "WSS", xlabel="#clusters", ylabel="WSS Score")
    plot_graph(incs, "inclines", xlabel="#clusters", ylabel="WSS graph inclines")


def main():
    #kmeans_model = PlaylistClusterer("data/playlist_features.csv")
    #get_cluster_sse_results(kmeans_model, res_file="sse_big_clusters.json", kmax=400, split_small_clusters=True, initial=10, interval=10)

    show_results("sse_big_clusters.json")
    #get_cluster_sse_results(kmeans_model, res_file="sse_reg.json", kmax=400, split_small_clusters=False, initial=10, interval=10)
    #show_results("sse_reg.json")


if __name__ == "__main__":
    main()
