import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Get coordinates satisfying a numpy condition as an array
def conditioned_coords_as_array(np_condition):
    rows, columns = np.where(np_condition)
    return np.asarray(list(zip(rows, columns)))

# Determines the number of clusters to detect in the saliency array using BIC
def get_number_clusters(saliency_array):
    n_components = range(1, 10)
    max_saliency = np.max(saliency_array)
    bic_input = conditioned_coords_as_array(saliency_array > max_saliency * 0.5)
    models = [GaussianMixture(n, covariance_type='full').fit(bic_input) for n in n_components]
    bic_scores = [model.bic(bic_input) for model in models]
    return n_components[np.argmin(bic_scores)]

# Optional function to show plot of clusters
def show_clusters(X, labels, centroids):
    plt.scatter(X[:, 1], X[:, 0], c=labels, cmap='viridis', s=50, alpha=0.8)
    plt.scatter(centroids[:, 1], centroids[:, 0], c='red', marker='X', s=200, label='Centroids')
    plt.title("clustering result")
    plt.show()

# Detect interest points in the saliency array using weighted KMeans clustering
# Returns a list of clusters containing:
# - the centroid coordinates
# - numpy array of pixel coordinates in the cluster
# - the average saliency of the cluster
def interest_clusters(saliency_array):
    clustering_input = conditioned_coords_as_array(saliency_array > 0)  # positive pixel coordinates
    clustering_weights = saliency_array[saliency_array > 0]             # use saliency as weight for each pixel
    kmeans = KMeans(get_number_clusters(saliency_array), n_init='auto')
    kmeans.fit(clustering_input, sample_weight=clustering_weights)
    show_clusters(clustering_input, kmeans.labels_, kmeans.cluster_centers_)

    results = []
    for cluster_index in range(len(kmeans.cluster_centers_)):
        cluster_data = {}
        cluster_data['centroid'] = kmeans.cluster_centers_[cluster_index]
        cluster_data['pixel_coords'] = clustering_input[kmeans.labels_ == cluster_index]
        cluster_data['avg_saliency'] = np.mean(clustering_weights[kmeans.labels_ == cluster_index])
        results.append(cluster_data)
    return results
