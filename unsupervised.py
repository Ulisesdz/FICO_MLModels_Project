# Importo las librerías principales
import pandas as pd
import matplotlib.pyplot as plt
import classification_models as clsf_modl
import numpy as np
import unsupervised_tools as UT
import seaborn as sns

# Importo los módulos de SKLearn
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.cluster.vq import vq

# Si se quiere graficar los diferentes clusters
plot = False

# Main
if __name__ == "__main__":
    
    input_vars = ['ExternalRiskEstimate','NetFractionRevolvingBurden','AverageMInFile','MSinceOldestTradeOpen','PercentTradesWBalance','PercentInstallTrades','NumSatisfactoryTrades','NumTotalTrades','PercentTradesNeverDelq','MSinceMostRecentInqexcl7days']
    # Cargo el archivo CSV sin los nans, estandarizado e imputado en un DataFrame
    df = pd.read_csv("Datasets/FICO_Dataset_modif.csv")
    df = df.iloc[:,1:]
    # Selecciono las columnas
    X_selected = df[input_vars]
    # Leo los archivos CSV
    X_train = pd.read_csv('Datasets/X_train_prueba.csv')
    X_test = pd.read_csv('Datasets/X_test_prueba.csv')
    y_train_df = pd.read_csv('Datasets/y_train_prueba.csv')
    y_test_df = pd.read_csv('Datasets/y_test_prueba.csv')
    # Convierto los DataFrames a Series
    y_train = y_train_df.iloc[:, 0]  # Selecciono la primera columna
    y_test = y_test_df.iloc[:, 0]  # Selecciono la primera columna
    X_train = X_train[input_vars]
    X_test = X_test[input_vars]
    # Los convierto en array para los modelos que lo requieran
    X_train_array = np.array(X_train)
    y_train_array = np.array(y_train)
    # Compruebo las dimensiones de los conjuntos de entrenamiento y prueba
    print("Dimensiones del conjunto de entrenamiento (X_train, y_train):", X_train.shape, y_train.shape)
    print("Dimensiones del conjunto de prueba (X_test, y_test):", X_test.shape, y_test.shape)
    print("="*50)
    
    # 1. PCA proceso de ajuste y análisis
    # Creo el modelo
    pca = clsf_modl.PCA(n_components=10)
    # Entreno el modelo PCA
    pca.fit(X_selected)
    # Transformo los datos en el espacio de las componentes principales
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    # Grafico la varianza explicada por componentes principales
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Número de componentes principales')
    plt.ylabel('Varianza explicada acumulada')
    plt.title('Varianza explicada acumulada por componentes principales')
    plt.grid(True)
    plt.show()
    
    # Hago PCA con SKLearn
    pca = PCA()
    pca.fit(X_selected)
    # Análisis de la varianza explicada
    explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = explained_variance.cumsum()
    # Visualización de la varianza explicada acumulada
    plt.plot(cumulative_explained_variance)
    plt.xlabel('Número de componentes principales')
    plt.ylabel('Varianza explicada acumulada')
    plt.title('Varianza explicada acumulada por componentes principales (SKLEARN)')
    plt.grid(True)
    plt.show()
    # Miro las importancias de las variables para los primeros 10 componentes principales
    n_components = min(10, X_selected.shape[1]) 
    importances = np.abs(pca.components_[:n_components])
    # Creo un dataframe con la informacion
    feature_names = X_selected.columns
    pca_feature_importance = pd.DataFrame(importances.T, columns=[f'Componente {i+1}' for i in range(n_components)], index=feature_names)
    # Lo grafico
    plt.figure(figsize=(12, 8))
    pca_feature_importance.plot(kind='bar', figsize=(12, 8))
    plt.xlabel('Variable')
    plt.ylabel('Importancia')
    plt.title('Importancia de cada variable en los primeros 10 componentes principales')
    plt.legend(title='Componente principal')
    plt.grid(True)
    plt.show()
    # Obtengo cuanto contribuye cada variable en cada componente principal
    contributions = np.abs(pca.components_) * np.sqrt(pca.explained_variance_.reshape(-1, 1))
    # Creo un dataframe con la informacion
    component_names = [f'Component {i+1}' for i in range(contributions.shape[0])]
    variable_names = X_selected.columns
    df_contributions = pd.DataFrame(contributions.T, index=variable_names, columns=component_names)
    # Lo grafico
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_contributions, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
    plt.xlabel('Componente Principal')
    plt.ylabel('Variable')
    plt.title('Contribución de cada variable en cada componente principal (Varianza explicada)')
    plt.show()
    # Creo el modelo con 6 componentes
    pca_6_componentes = PCA(n_components=6)
    pca_6_componentes.fit(X_selected)
    # Proyecto los datos al espacio de 6 componentes principales
    datos_proyectados = pca_6_componentes.transform(X_selected)
    df_proyectados = pd.DataFrame(datos_proyectados, columns=['Componente Principal '+str(i) for i in range(1, 7)])
    # Crear el gráfico de pares
    sns.pairplot(df_proyectados)
    plt.suptitle('Datos proyectados en el espacio de 6 componentes principales', y=1.02)
    plt.show()
    
    
    # 2. Identificación y ajuste de modelos de clustering
    
    # Inicializo el Scaler
    scaler = StandardScaler()
    X_transformed = scaler.fit_transform(X_selected)
    
    ## Hierarchical clustering -------------------------------------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Cambio a una sola fila con tres subplots
    # Los métodos de linkage que voy a usar
    linkage_methods = ['single', 'complete', 'average']
    # Genero los dendogramas
    for i, method in enumerate(linkage_methods):
        linked = linkage(X_transformed, method)
        dendrogram(linked,
                orientation='top',
                labels=df.index,
                truncate_mode='level',  # Tipo de truncamiento
                p=5,  # Número de hojas para mostrar
                above_threshold_color='y',
                color_threshold=100,
                distance_sort='descending',
                show_leaf_counts=True,
                ax=axs[i])  # Cambio a axs[i] para una fila

        axs[i].set_title(f'Dendrogram with {method.capitalize()} Linkage')
    # Muestro la gráfica
    plt.tight_layout()
    plt.show()
    linked = linkage(X_transformed, 'ward')

    plt.figure(figsize=(10, 7))
    dendrogram(linked,
                orientation='top',
                labels=df.index,
                truncate_mode= 'level', # Type of truncation, in this case number of levels. A “level” includes all nodes with p merges from the last merge.
                p = 5, # Number of leaves to show
                above_threshold_color='y',
                color_threshold=100,
                distance_sort='descending',
                show_leaf_counts=True)
    plt.gca().set_title('Dendrogram using level truncation method')
    plt.show()
    ## Silhouette method ---------------s
    n_clusters = 2
    cluster_labels = cut_tree(linked, n_clusters=n_clusters)
    silhouette_avg = silhouette_score(X_transformed, cluster_labels)
    print("The average silhouette_score is :", silhouette_avg)
    # Muestro los clusters
    UT.plot_clusters(df, cluster_labels, alpha_curves=0.3, figsize=(9,6))
    
    
    ## K_means and silhouette method simultaneously with different number of clusters
    range_n_clusters = list(range(2,10))
    SSQ = []
    sil_avg = []
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X_transformed)
        _ , err = vq(X_transformed, clusterer.cluster_centers_)
        SSQ.append(np.sum(err**2))
        # Obtengo la silueta
        sil_avg.append(silhouette_score(X_transformed, cluster_labels))
        if plot:
            UT.plot_clusters(df, cluster_labels, centers=clusterer.cluster_centers_, figsize=(6,4))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(range_n_clusters,SSQ, marker='o')
    ax1.set_title("Sum of Squares")
    ax1.set_xlabel("Number of clusters")
    ax2.plot(range_n_clusters,sil_avg, marker='o')
    ax2.set_title("Silhouette values")
    ax2.set_xlabel("Number of clusters")
    plt.show()
    
    clusterer = KMeans(n_clusters=2, random_state=10)
    cluster_knn = clusterer.fit_predict(X_transformed)
    print('Clusters specifying number of clusters:')
    unique, counts = np.unique(cluster_knn, return_counts=True)
    print(pd.DataFrame(np.asarray(counts), index=unique, columns=['# Samples']))
    # Silueta
    silhouette_avg = silhouette_score(X_transformed, cluster_knn)
    print("The average silhouette_score is :", silhouette_avg)
    UT.plot_clusters(df, cluster_knn, centers= scaler.inverse_transform(clusterer.cluster_centers_), figsize=(10,6))
    
    
    ## Gaussian Mixture Models -------------------------------------------------------------------------------------------------------
    range_n_clusters = list(range(2,10))
    SSQ = []
    sil_avg = []
    cv_type = 'full'
    for n_clusters in range_n_clusters:

        clusterer = GaussianMixture(n_components=n_clusters,
                                covariance_type=cv_type,
                                random_state=10)
        cluster_labels = clusterer.fit_predict(X_transformed)

        _ , err = vq(X_transformed, clusterer.means_)
        SSQ.append(np.sum(err**2))
        # Obtengo la silueta
        sil_avg.append(silhouette_score(X_transformed, cluster_labels))
        if plot:
            UT.plot_clusters(df, cluster_labels, centers=clusterer.means_, figsize=(6,4))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(range_n_clusters,SSQ, marker='o')
    ax1.set_title("Sum of Squares")
    ax1.set_xlabel("Number of clusters")
    ax2.plot(range_n_clusters,sil_avg, marker='o')
    ax2.set_title("Silhouette values")
    ax2.set_xlabel("Number of clusters")
    plt.show()
    
    clusterer_GMM = GaussianMixture(n_components=2,
                                covariance_type='full',
                                random_state=10)
    cluster_GMM = clusterer_GMM.fit_predict(X_transformed)
    print('Clusters specifying number of clusters:')
    unique, counts = np.unique(cluster_knn, return_counts=True)
    print(pd.DataFrame(np.asarray(counts), index=unique, columns=['# Samples']))
    # Silueta
    silhouette_avg = silhouette_score(X_transformed, cluster_knn)
    print("The average silhouette_score is :", silhouette_avg)
    UT.plot_clusters(df, cluster_GMM, centers= scaler.inverse_transform(clusterer_GMM.means_), figsize=(10,6))
    
    
    # Muestro los datos proyectados en 2 componentes principales frente a clustering con 2 clusters
    df = pd.read_csv("Datasets/FICO_Dataset_modif.csv")
    X = df.drop("RiskPerformance", axis=1)  
    y = df["RiskPerformance"]

    # Aplico PCA
    pca = clsf_modl.PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)

    # Grafico los puntos
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='red', label='RiskPerformance: 0')
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='blue', label='RiskPerformance: 1')
    plt.title('Nube de Puntos con PCA')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.show()
    
    # KMeans clustering
    clusterer = KMeans(n_clusters=2, random_state=10)
    cluster_knn = clusterer.fit_predict(X)

    # Creo la figura y ejes para subgráficos
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Grafico la nube de puntos de PCA en el primer subgráfico
    axs[0].scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='red', label='RiskPerformance: 0')
    axs[0].scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='blue', label='RiskPerformance: 1')
    axs[0].set_title('Nube de Puntos con PCA')
    axs[0].set_xlabel('Componente Principal 1')
    axs[0].set_ylabel('Componente Principal 2')
    axs[0].legend()

    # Grafico la nube de puntos de KMeans en el segundo subgráfico
    axs[1].scatter(X_pca[cluster_knn == 0, 0], X_pca[cluster_knn == 0, 1], color='green', label='Cluster 1')
    axs[1].scatter(X_pca[cluster_knn == 1, 0], X_pca[cluster_knn == 1, 1], color='orange', label='Cluster 2')
    axs[1].set_title('Nube de Puntos con Clustering KMeans')
    axs[1].set_xlabel('Componente Principal 1')
    axs[1].set_ylabel('Componente Principal 2')
    axs[1].legend()

    # Muestro los gráficos
    plt.tight_layout()
    plt.show()