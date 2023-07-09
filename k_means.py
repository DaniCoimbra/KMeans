import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math

def calculateSilhouetteScore(clusters):
    datapoints = []
    labels = []
    for centroid, points in clusters.items():
        datapoints.extend(points)
        labels.extend([centroid] * len(points))

    # Encode labels as integers
    label_encoder = {centroid: index for index, centroid in enumerate(clusters.keys())}
    encoded_labels = [label_encoder[label] for label in labels]

    # Calculate a(i) and b(i) for each data point
    a_values = []
    b_values = []
    for i in range(len(datapoints)):
        point = datapoints[i]
        label = encoded_labels[i]

        # Calculate a(i) - average distance within the same cluster
        a_sum = 0
        a_count = 0
        for j in range(len(datapoints)):
            if encoded_labels[j] == label and i != j:
                a_sum += math.sqrt(sum((point[k] - datapoints[j][k])**2 for k in range(len(point))))
                a_count += 1
        a_i = a_sum / a_count if a_count > 0 else 0
        a_values.append(a_i)

        # Calculate b(i) - average distance to the nearest neighboring cluster
        b_values.append(math.inf)
        for k in range(len(clusters)):
            if k != label:
                b_sum = 0
                b_count = 0
                for j in range(len(datapoints)):
                    if encoded_labels[j] == k:
                        b_sum += math.sqrt(sum((point[l] - datapoints[j][l])**2 for l in range(len(point))))
                        b_count += 1
                b_i = b_sum / b_count if b_count > 0 else 0
                b_values[i] = min(b_i, b_values[i])

    # Calculate silhouette coefficients
    silhouette_coefficients = [(b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) != 0 else 0
                               for a_i, b_i in zip(a_values, b_values)]

    # Calculate silhouette score
    silhouette_score = sum(silhouette_coefficients) / len(silhouette_coefficients)

    return silhouette_score


def calculate_k(dataset):
    # Convert the dataset to numpy array
    X = np.array(dataset)

    # Initialize variables to store the best silhouette score and corresponding k value
    best_score = -1
    best_k = -1
    n = math.ceil(np.log2(len(dataset))*np.log10(len(dataset)))

    # Apply k-means for k values from 2 to min(11, number of samples - 1)
    for k in range(2, min(12, len(dataset))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init= n)
        kmeans.fit(X)
        labels = kmeans.labels_
        score = silhouette_score(X, labels)

        # Check if the current k value provides a better silhouette score
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

def genereateCentroids(best_k, dataset):
    kmeans = KMeans(n_clusters=best_k, n_init= 'auto')
    # Step 2b: Fit the algorithm to the dataset
    kmeans.fit(dataset)
    # Step 2c: Calculate centroids
    centroids = kmeans.cluster_centers_
    centroids = [tuple(centroid) for centroid in centroids]
    clusters = {i: [] for i in centroids}
    for datapoint in dataset:
        distance = math.inf
        c = 0
        for centroid in centroids:
            if math.sqrt((datapoint[0]-centroid[0])**2 + (datapoint[1]-centroid[1])**2) < distance:
                distance = math.sqrt((datapoint[0]-centroid[0])**2 + (datapoint[1]-centroid[1])**2)
                c = centroid
        clusters[c].append(datapoint)
    #print(calculateSilhouetteScore(clusters))
    return clusters

def plotGraph(clusters):
    x_values = []
    y_values = []
    hue_values = []
    for group, points in clusters.items():
        x, y = zip(*points)
        x_values.extend(x)
        y_values.extend(y)
        hue_values.extend([group] * len(points))

    # Create a DataFrame from the extracted values
    df = pd.DataFrame({'X': x_values, 'Y': y_values, 'Group': hue_values})

    # Plot the values using seaborn
    sns.scatterplot(data=df, x='X', y='Y', hue='Group', palette='Set1')

    # Display the plot
    plt.show()

