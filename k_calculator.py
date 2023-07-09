import numpy as np
import math
import random
import copy
from sklearn.metrics import silhouette_score


def calculateSilhouetteScore(clusters):
    # Flatten the cluster dictionary into lists of points and labels
    datapoints = []
    labels = []
    for centroid, points in clusters.items():
        datapoints.extend(points)
        labels.extend([centroid] * len(points))

    # Encode labels as integers
    label_encoder = {centroid: index for index, centroid in enumerate(clusters.keys())}
    encoded_labels = np.array([label_encoder[label] for label in labels])

    # Convert datapoints to a NumPy array
    datapoints = np.array(datapoints)

    # Calculate silhouette score
    silhouette = silhouette_score(datapoints, encoded_labels)

    return silhouette


def updateClusters(centroids, dataset):
    clusters = {centroid: set() for centroid in centroids}
    for datapoint in dataset:
        distance = math.inf
        c = None
        for centroid in centroids:
            dist = math.sqrt((datapoint[0] - centroid[0]) ** 2 + (datapoint[1] - centroid[1]) ** 2)
            if dist < distance:
                distance = dist
                c = centroid
        clusters[c].add(datapoint)
    return clusters


def calculateCentroids(dataset, k, max_iterations=100):
    centroids = []
    centroids.append(random.choice(tuple(dataset)))

    for _ in range(1, k):
        distances = [min(math.dist(datapoint, centroid) ** 2 for centroid in centroids) for datapoint in dataset]
        new_centroid = dataset[distances.index(max(distances))]
        centroids.append(new_centroid)

    clusters = updateClusters(centroids, dataset)

    previous_clusters = None
    iteration = 0
    while previous_clusters != clusters and iteration < max_iterations:
        previous_clusters = copy.deepcopy(clusters)

        for dp in dataset:
            smallestDist = math.inf
            smallestC = None
            for c in centroids:
                if dp in clusters[c]:
                    clusters[c].remove(dp)
                    if len(clusters[c]) == 0:
                        clusters[c].add(dp)
                        continue
                clusters[c].add(dp)
                localDist = sum(math.dist(c, point) for point in clusters[c])
                clusters[c].remove(dp)
                if smallestDist > localDist:
                    smallestDist = localDist
                    smallestC = c
            if smallestC is not None:
                clusters[smallestC].add(dp)

        centroids = [
            (sum(point[0] for point in cluster) / len(cluster), sum(point[1] for point in cluster) / len(cluster))
            for cluster in clusters.values()]
        clusters = updateClusters(centroids, dataset)

        iteration += 1

    return clusters

def silhouetteScores(dataset):
    optimalS = 0
    optimalK = 0

    for k in range(2,min(12,len(dataset))):
        localOptimalS = 0
        for i in range(math.ceil(np.log2(len(dataset))*np.log10(len(dataset)))):
            clusters = calculateCentroids(dataset, k)
            s = calculateSilhouetteScore(clusters)
            if s > localOptimalS:
                localOptimalS = s
        print("For a k value of %d, the silhouetteScore is: "%k,end=" ")
        print(localOptimalS)
        if localOptimalS > optimalS:
                optimalS = localOptimalS
                optimalK = k
    print("The optimal k value is: %d, with a silhouetteScore of: " %optimalK ,end=" ")
    print(optimalS)

