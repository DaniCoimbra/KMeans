import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generateDataset(dataset_size):
    cluster_centers = [(20, 20), (80, 80), (50, 60), (50, 20), (30, 70), (70, 30)]
    dataset = []

    # Generate points around cluster centers
    for _ in range(dataset_size):
        cluster_center = random.choice(cluster_centers)
        x = random.gauss(cluster_center[0]*2, random.uniform(5, 15))
        y = random.gauss(cluster_center[1]*2, random.uniform(5, 15))
        dataset.append((x, y))
    print(dataset)
    return dataset

def plotDataset(dataset):
    x_values, y_values = zip(*dataset)

    # Create a DataFrame
    df = pd.DataFrame({'X': x_values, 'Y': y_values})

    # Plot the scatter graph
    sns.scatterplot(data=df, x='X', y='Y')

    # Display the plot
    plt.show()
