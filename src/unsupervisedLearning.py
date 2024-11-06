import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

def kmeans_clustering(X):
    print("\nAnalisi di clustering non supervisionato:")
    print("Scopo: Identificare potenziali sottotipi di cancro al seno basati sulle caratteristiche del tumore.")
    
    X_scaled = X

    optimal_k = find_optimal_k_bic(X_scaled)

    kmeans = KMeans(n_clusters=optimal_k, n_init=10, init='random', random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    X_clustered = X.copy()
    X_clustered['Cluster'] = cluster_labels

    visualize_clusters(X_clustered, optimal_k)
    visualize_clusters_3d(X_scaled, cluster_labels)
    analyze_clusters(X_clustered, kmeans)

    return X_clustered

def find_optimal_k_bic(X_scaled):
    max_clusters = 10
    min_clusters = 2
    bic_scores = []

    for k in range(min_clusters, max_clusters + 1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X_scaled)
        bic_scores.append(gmm.bic(X_scaled))

    optimal_k = bic_scores.index(min(bic_scores)) + min_clusters

    plt.plot(range(min_clusters, max_clusters + 1), bic_scores, 'bx-')
    plt.scatter(optimal_k, bic_scores[optimal_k - min_clusters], c='red', label=f'Miglior k: {optimal_k}')
    plt.xlabel('Numero di Cluster (k)')
    plt.ylabel('BIC Score')
    plt.title('Bayesian Information Criterion')
    plt.legend()
    plt.show()

    print(f"Numero ottimale di cluster: {optimal_k}")
    return optimal_k

def visualize_clusters(X_clustered, n_clusters):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_clustered.drop('Cluster', axis=1))
    
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=X_clustered['Cluster'], cmap='viridis')
    plt.colorbar(scatter)
    plt.xlabel('Prima Componente Principale')
    plt.ylabel('Seconda Componente Principale')
    plt.title('Visualizzazione dei cluster K-Means (PCA)')
    plt.show()

def visualize_clusters_3d(X_scaled, cluster_labels):
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=cluster_labels, cmap='viridis')
    plt.colorbar(scatter)
    ax.set_xlabel('Prima Componente Principale')
    ax.set_ylabel('Seconda Componente Principale')
    ax.set_zlabel('Terza Componente Principale')
    plt.title('Visualizzazione 3D dei cluster K-Means (PCA)')
    plt.show()

def analyze_clusters(X_clustered, kmeans):
    cluster_means = X_clustered.groupby('Cluster').mean()
    print("\nCaratteristiche medie per cluster:")
    print(cluster_means)

    feature_importance = pd.DataFrame(kmeans.cluster_centers_, 
                                      columns=X_clustered.columns[:-1])
    print("\nFeature più importanti per ciascun cluster:")
    for cluster in range(len(feature_importance)):
        print(f"\nCluster {cluster}:")
        print(feature_importance.iloc[cluster].nlargest(5))

    print("\nInterpretazione dei cluster:")
    cluster_interpretations = [
        "Potenziali tumori di grandi dimensioni, possibilmente ad alta malignità",
        "Tumori di dimensioni medie", 
        "Tumori di dimensioni più piccole, potenzialmente meno aggressivi",
        "Tumori di dimensioni medio-grandi"
    ]
    cluster_colors = ['skyblue', 'lightcoral', 'yellow', 'purple']
    for cluster in range(len(feature_importance)):
        print(f"\nCluster {cluster} (colore: {cluster_colors[cluster]}): {cluster_interpretations[cluster]}")
        top_features = feature_importance.iloc[cluster].nlargest(3)
        for feature, value in top_features.items():
            print(f"- Alto valore di {feature}: {value:.2f}")
