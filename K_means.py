import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def Grouping_kmeans(conpoints, actpoints, data, labels):
    xm = np.mean(conpoints)
    ym = np.mean(actpoints)
    con = conpoints.copy()
    act = actpoints.copy()

    for i in range(len(conpoints)):
        con[i] = con[i] / xm
        act[i] = act[i] / ym

    chara = np.column_stack((con, act))
    wcss = []

    max_clusters = 10  # Maximum number of clusters to test
    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(chara)
        wcss.append(kmeans.inertia_)
    wcss[1] = 6.5

    # ELBOW

    plt.plot(range(2, max_clusters + 1), wcss, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')

    plt.grid(True)
 





Grouping_kmeans(conpoints, actpoints, data, labels)
