import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import json
def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax+1, 10):
        print(f"k={k}")
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += np.linalg.norm(points[i]-curr_center, ord=2)
        sse.append(curr_sse)
        with open("sse.json", "w") as f:
            json.dump(sse, f)

    plt.show(sse)

def main():
    #points_df = pd.read_csv("data/user_features.csv")
    #points = np.array(points_df)
    #calculate_WSS(points, 400)

    with open("sse.json") as f:
        sse = json.load(f)

    for i, s in enumerate(sse):
        print(f"{i} : {s}")

    plt.plot(sse)
    plt.show()

    #a = np.argsort(sse)

    #b = 5




    pass

if __name__ == "__main__":
    main()
