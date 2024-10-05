import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def evaluate_classification_metrics(y_true, y_pred, positive_label, auc):
    """
    Calculate various evaluation metrics for a classification model.

    Args:
        y_true (array-like): True labels of the data.
        positive_label: The label considered as the positive class.
        y_pred (array-like): Predicted labels by the model.
        auc (float): Area under the ROC Curve

    Returns:
        dict: A dictionary containing various evaluation metrics.

    Metrics Calculated:
        - Confusion Matrix: [TN, FP, FN, TP]
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Precision: TP / (TP + FP)
        - Recall (Sensitivity): TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    """
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    y_pred_mapped = np.array([1 if label == positive_label else 0 for label in y_pred])

    # Confusion Matrix
    tp = sum((y_pred_mapped == 1) & (y_true_mapped == 1))
    tn = sum((y_pred_mapped == 0) & (y_true_mapped == 0))
    fp = sum((y_pred_mapped == 1) & (y_true_mapped == 0))
    fn = sum((y_pred_mapped == 0) & (y_true_mapped == 1))

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Precision
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0

    # Recall (Sensitivity)
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return {
        'Confusion Matrix': [tn, fp, fn, tp],
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1 Score': f1,
        'AUC ROC': auc        
    }

def plot_calibration_curve(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot a calibration curve to evaluate the accuracy of predicted probabilities.

    This function creates a plot that compares the mean predicted probabilities
    in each bin with the fraction of positives (true outcomes) in that bin.
    This helps assess how well the probabilities are calibrated.

    Args:
        y_true (array-like): True labels of the data.
        y_probs (array-like): Predicted probabilities for the positive class (1).
        positive_label: The label considered as the positive class.
        n_bins (int, optional): Number of equally spaced bins to use for calibration. Defaults to 10.

    Returns:
        None: This function plots the calibration curve and does not return any value.

    """
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    true_proportions = np.zeros(n_bins)

    for i in range(n_bins):
        indices = (y_probs >= bins[i]) & (y_probs < bins[i+1])
        if np.sum(indices) > 0:
            true_proportions[i] = np.mean(y_true_mapped[indices])

    plt.plot(bin_centers, true_proportions, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.show()

def plot_probability_histograms(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot probability histograms for the positive and negative classes separately.

    This function creates two histograms showing the distribution of predicted
    probabilities for each class. This helps in understanding how the model
    differentiates between the classes.

    Args:
        y_true (array-like): True labels of the data.
        y_probs (array-like): Predicted probabilities for the positive class.
        positive_label: The label considered as the positive class.
        n_bins (int, optional): Number of bins for the histograms. Defaults to 10.

    Returns:
        None: This function plots the histograms and does not return any value.

    """
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])

    plt.figure(figsize=(12, 6))

    # Histogram for positive class
    plt.subplot(1, 2, 2)
    plt.hist(y_probs[y_true_mapped == 1], bins=n_bins, color='green', alpha=0.7)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Probability Histogram (Positive Class)')

    # Histogram for negative class
    plt.subplot(1, 2, 1)
    plt.hist(y_probs[y_true_mapped == 0], bins=n_bins, color='red', alpha=0.7)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Probability Histogram (Negative Class)')

    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_probs, positive_label):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    The ROC curve is a graphical representation of the diagnostic ability of a binary
    classifier system as its discrimination threshold is varied. It plots the True Positive
    Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

    Args:
        y_true (array-like): True labels of the data.
        y_probs (array-like): Predicted probabilities for the positive class (1).
        positive_label: The label considered as the positive class.

    Returns:
        float: The Area under the ROC curve (AUC ROC).

    """
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])

    thresholds = np.linspace(0, 1, 11)
    tpr = []  # True Positive Rate
    fpr = []  # False Positive Rate

    # Inicializo el área en 0
    auc_roc = 0.0

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        tp = np.sum((y_true_mapped == 1) & (y_pred == 1))
        fp = np.sum((y_true_mapped == 0) & (y_pred == 1))
        fn = np.sum((y_true_mapped == 1) & (y_pred == 0))
        tn = np.sum((y_true_mapped == 0) & (y_pred == 0))

        tpr_value = tp / (tp + fn) if tp + fn != 0 else 0
        fpr_value = fp / (fp + tn) if fp + tn != 0 else 0

        tpr.append(tpr_value)
        fpr.append(fpr_value)
        
        # Solo calculo AUC si hay más de un punto en la curva ROC
        if len(tpr) > 1: 
            auc_roc += (fpr[-1] - fpr[-2]) * (tpr[-1] + tpr[-2]) / 2

    tpr.append(0)
    fpr.append(0)

    plt.plot(fpr, tpr, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve\nAUC: {:.2f}'.format(abs(auc_roc)))
    plt.show()
    return abs(auc_roc)

def classification_report(y_true, y_probs, positive_label, threshold=0.5, n_bins=10):
    """
    Create a classification report using the auxiliary functions developed during Lab2_1

    Args:
        y_true (array-like): True labels of the data.
        y_probs (array-like): Predicted probabilities for the positive class.
        positive_label: The label considered as the positive class.
        threshold (float): Threshold to transform probabilities to predictions. Defaults to 0.5.
        n_bins (int, optional): Number of bins for the histograms and equally spaced 
                                bins to use for calibration. Defaults to 10.

    Returns:
        dict: A dictionary containing various evaluation metrics.

    Metrics Calculated:
        - Confusion Matrix: [TN, FP, FN, TP]
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Precision: TP / (TP + FP)
        - Recall (Sensitivity): TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)

    """
    plot_calibration_curve(y_true, y_probs, positive_label, n_bins)
    plot_probability_histograms(y_true, y_probs, positive_label, n_bins)
    auc = plot_roc_curve(y_true, y_probs, positive_label)
    return evaluate_classification_metrics(y_true, (y_probs > threshold).astype(int), positive_label, auc = auc)

def train_test_split(X, y, test_size=0.2, stratify=None, random_state=None):
    """
    Splits arrays or matrices into random train and test subsets. This function demonstrates how to 
    divide a dataset into training and testing sets, optionally stratifying the samples and ensuring 
    reproducibility with a random state.

    Parameters:
    - X (np.ndarray): Input features matrix, where rows represent samples and columns represent features.
    - y (np.ndarray): Target labels array, aligned with the samples in X.
    - test_size (float or int): Determines the size of the test set. If float, it represents a proportion 
                                of the dataset; if int, it specifies the number of samples.
    - stratify (np.ndarray): If provided, the function will ensure the class proportions in train and test 
                             sets mirror those of the provided array, typically the target labels array.
    - random_state (int): Seed for the random number generator to ensure reproducible splits.

    Returns:
    - X_train, X_test, y_train, y_test: Arrays containing the split of features and labels into training and 
                                        test sets.
    """
    
    # Set the seed for reproducibility
    if random_state:
        np.random.seed(random_state)

    # Determine the number of samples to allocate to the test set
    n_samples = X.shape[0]
    if isinstance(test_size, float):
        n_test = int(n_samples * test_size)
    else:
        n_test = test_size
    n_train = n_samples - n_test

    # Create an array of indices and shuffle if not stratifying
    indices = np.arange(n_samples)
    if stratify is None:
        np.random.shuffle(indices)
    else:
        # For stratified splitting, determine the distribution of classes
        unique_classes, y_indices = np.unique(stratify, return_inverse=True)
        class_counts = np.bincount(y_indices)
        test_counts = np.round(class_counts * test_size).astype(int)

        # Allocate indices to train and test sets preserving class distribution
        train_indices, test_indices = [], []
        for class_index in range(len(unique_classes)):
            class_indices = indices[y_indices == class_index]
            np.random.shuffle(class_indices)
            boundary = test_counts[class_index]
            test_indices.extend(class_indices[:boundary])
            train_indices.extend(class_indices[boundary:])

        # Concatenate indices to form the final split
        indices = train_indices + test_indices

    # Use the indices to partition the dataset
    X_train = X[indices[:n_train]]
    X_test = X[indices[n_train:]]
    y_train = y[indices[:n_train]]
    y_test = y[indices[n_train:]]

    return X_train, X_test, y_train, y_test
    

def cross_validation(model, X, y, nFolds):
    """
    Perform cross-validation on a given machine learning model to evaluate its performance.
    
    This function manually implements n-fold cross-validation if a specific number of folds is provided.
    If nFolds is set to -1, Leave One Out (LOO) cross-validation is performed instead, which uses each
    data point as a single test set while the rest of the data serves as the training set.
    
    Parameters:
    - model: scikit-learn-like estimator
        The machine learning model to be evaluated. This model must implement the .fit() and .score() methods
        similar to scikit-learn models.
    - X: array-like of shape (n_samples, n_features)
        The input features to be used for training and testing the model.
    - y: array-like of shape (n_samples,)
        The target values (class labels in classification, real numbers in regression) for the input samples.
    - nFolds: int
        The number of folds to use for cross-validation. If set to -1, LOO cross-validation is performed.
    
    Returns:
    - mean_score: float
        The mean score across all cross-validation folds.
    - std_score: float
        The standard deviation of the scores across all cross-validation folds, indicating the variability
        of the score across folds.
    
    """
    if nFolds == -1:
        # Implement Leave One Out CV
        nFolds = X.shape[0]
    
    # Calculate fold_size based on the number of folds
    fold_size = X.shape[0] // nFolds

    # Initialize a list to store the accuracy values of the model for each fold
    accuracy_scores = []
    
    for i in range(nFolds):
        # Generate indices of samples for the validation set for the fold
        valid_indices = list(range(i*fold_size, (i+1)*fold_size))
        
        # Generate indices of samples for the training set for the fold
        train_indices = list(set(range(X.shape[0])) - set(valid_indices))
        
        # Split the dataset into training and validation
        X_train, X_valid = X[train_indices], X[valid_indices]
        y_train, y_valid = y[train_indices], y[valid_indices]
        
        # Train the model with the training set
        model.fit(X_train, y_train)
        
        # Predict the labels of the validation set using the trained model
        y_pred = model.predict(X_valid)
        
        # Calculate the accuracy of the model with the validation set and store it in accuracy_scores
        accuracy_scores.append(np.mean(y_pred == y_valid))
    
    # Return the mean and standard deviation of the accuracy_scores 
    return np.mean(accuracy_scores), np.std(accuracy_scores)

def cross_validation_decisionTrees(model, X, y, nFolds):
    """
    Perform cross-validation on a given a DecisionTree to evaluate its performance.
    
    This function manually implements n-fold cross-validation if a specific number of folds is provided.
    If nFolds is set to -1, Leave One Out (LOO) cross-validation is performed instead, which uses each
    data point as a single test set while the rest of the data serves as the training set.
    
    Parameters:
    - model: DecisionTree model
    - X: array-like of shape (n_samples, n_features)
        The input features to be used for training and testing the model.
    - y: array-like of shape (n_samples,)
        The target values (class labels in classification, real numbers in regression) for the input samples.
    - nFolds: int
        The number of folds to use for cross-validation. If set to -1, LOO cross-validation is performed.
    
    Returns:
    - mean_score: float
        The mean score across all cross-validation folds.
    - std_score: float
        The standard deviation of the scores across all cross-validation folds, indicating the variability
        of the score across folds.
    
    """
    if nFolds == -1:
        # Implement Leave One Out CV
        nFolds = X.shape[0]
    
    # Calculate fold_size based on the number of folds
    fold_size = X.shape[0] // nFolds

    # Initialize a list to store the accuracy values of the model for each fold
    accuracy_scores = []
    
    for i in range(nFolds):
        # Generate indices of samples for the validation set for the fold
        valid_indices = list(range(i*fold_size, (i+1)*fold_size))
        
        # Generate indices of samples for the training set for the fold
        train_indices = list(set(range(X.shape[0])) - set(valid_indices))
        
        # Split the dataset into training and validation
        X_train, X_valid = X[train_indices], X[valid_indices]
        y_train, y_valid = y[train_indices], y[valid_indices]
        
        # Train the model with the training set
        model.fit(X_train, y_train)
        
        # Predict the labels of the validation set using the trained model
        y_pred = model.predict(X_valid)
        
        # Calculo la accuracy
        metricas = evaluate_classification_metrics(y_valid, (np.array(y_pred) > 0.5).astype(int), 1, auc=None)
        accuracy = metricas["Accuracy"]
        accuracy_scores.append(accuracy)
    
    # Return the mean and standard deviation of the accuracy_scores 
    return np.mean(accuracy_scores), np.std(accuracy_scores)

def accuracy_score(y_true, y_pred):
    """
    Calcula la precisión del conjunto de etiquetas y_pred en comparación con las etiquetas verdaderas y_true.

    Parameters:
    y_true : array-like de shape (n_samples,)
        Las etiquetas verdaderas.
        
    y_pred : array-like de shape (n_samples,)
        Las etiquetas predichas.

    Returns:
    accuracy : float
        La precisión del clasificador.
    """
    # Verifica que las longitudes de y_true y y_pred sean iguales
    if len(y_true) != len(y_pred):
        raise ValueError("Las longitudes de y_true y y_pred deben ser iguales.")
    
    # Calcula la precisión
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    total = len(y_true)
    accuracy = correct / total
    
    return accuracy

def escribir_resultados_en_csv(nombre_archivo, nombre_modelo, metricas):
    # Creo un DataFrame para los datos del modelo
    df_modelo = pd.DataFrame([metricas])
    df_modelo['Modelo'] = nombre_modelo
    # Separo la matriz de confusión en cuatro columnas
    df_modelo['TP'] = metricas['Confusion Matrix'][0]
    df_modelo['FP'] = metricas['Confusion Matrix'][1]
    df_modelo['FN'] = metricas['Confusion Matrix'][2]
    df_modelo['TN'] = metricas['Confusion Matrix'][3]
    # Elimino la columna de matriz de confusión
    del df_modelo['Confusion Matrix']
    # Reordeno las columnas para que el nombre del modelo esté al principio
    columnas = ['Modelo', 'TP', 'FP', 'FN', 'TN'] + [col for col in df_modelo.columns if col not in ['Modelo', 'TP', 'FP', 'FN', 'TN']]
    df_modelo = df_modelo[columnas]
    
    # Guardo los datos en el archivo CSV
    # Si no existe el archivo lo creo
    if not os.path.exists(nombre_archivo):
        df_modelo.to_csv(nombre_archivo, mode='w', index=False)
    # Si existe, lo abro y lo modifico
    else:
        df_existente = pd.read_csv(nombre_archivo)
        # Si el modelo ya existe en el archivo, reescribo los datos
        if nombre_modelo in df_existente['Modelo'].values:
            df_existente = df_existente[df_existente['Modelo'] != nombre_modelo]  # Elimino el modelo existente
            df_existente = pd.concat([df_existente, df_modelo], ignore_index=True)  # Añado el nuevo modelo
            df_existente.to_csv(nombre_archivo, mode='w', index=False)
        else:
            # Si el modelo no existe, añado los datos
            df_modelo.to_csv(nombre_archivo, mode='a', index=False, header=False)
            
            
def calcular_importancia_variables_array(modelo, X, y):
    # Obtengo la precisión inicial del modelo con todas las características
    y_pred_proba_inicial = modelo.predict_proba(X)
    accuracy_inicial = accuracy_score(y, y_pred_proba_inicial[:, 1])
    
    importancia_variables = []
    
    # Itero sobre cada característica en el conjunto de datos
    for i in range(X.shape[1]):
        # Creo una copia del conjunto de datos con la característica i eliminada
        X_sin_variable = np.delete(X, i, axis=1)
        
        # Entreno el modelo sin la característica i
        modelo.fit(X_sin_variable, y)
        
        # Calculo la precisión del modelo sin la característica i
        y_pred_proba_sin_variable = modelo.predict_proba(X_sin_variable)
        accuracy_sin_variable = accuracy_score(y, y_pred_proba_sin_variable[:, 1])
        
        # Calculo el cambio en la precisión
        importancia = accuracy_inicial - accuracy_sin_variable
        
        importancia_variables.append(importancia)
    
    return importancia_variables


def calcular_importancia_variables(modelo, X, y):
    # Obtengo las probabilidades iniciales del modelo con todas las características
    y_pred_proba_inicial = modelo.predict_proba(X)
    metricas = evaluate_classification_metrics(y, (np.array(y_pred_proba_inicial) > 0.5).astype(int), 1, auc=None)
    accuracy_inicial = metricas["Accuracy"]
    
    importancia_variables = []
    
    # Itero sobre cada característica en el conjunto de datos
    for i in range(X.shape[1]):
        # Creo una copia del conjunto de datos con la característica i eliminada
        X_sin_variable = np.delete(X, i, axis=1)
        
        # Entreno el modelo sin la característica i
        modelo.fit(X_sin_variable, y)
        
        # Calculo las probabilidades del modelo sin la característica i
        y_pred_proba_sin_variable = modelo.predict_proba(X_sin_variable)
        metricas = evaluate_classification_metrics(y, (np.array(y_pred_proba_sin_variable) > 0.5).astype(int), 1, auc=None)
        accuracy_sin_variable = metricas["Accuracy"]
        
        # Calculo el cambio en la precisión
        importancia = accuracy_inicial - accuracy_sin_variable
        
        importancia_variables.append(importancia)
    
    return importancia_variables


def calcular_importancia_variables_arboles(modelo, X, y, random_seed=None):
    # Convierto X a un arreglo NumPy
    X_array = np.array(X)
    
    # Fijo la semilla aleatoria si se proporciona
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Calculo la precisión inicial
    y_pred_inicial = modelo.predict(X_array)
    accuracy_inicial = accuracy_score(y, y_pred_inicial)
    
    importancia_variables = []
    
    # Itero sobre cada característica en el conjunto de datos
    for i in range(X_array.shape[1]):
        # Copio el conjunto de datos con la característica i permutada aleatoriamente
        X_permutado = X_array.copy()
        np.random.shuffle(X_permutado[:, i])
        
        # Calculo la precisión con la característica permutada
        y_pred_permutado = modelo.predict(X_permutado)
        accuracy_permutado = accuracy_score(y, y_pred_permutado)
        
        # Calculo la importancia como la diferencia en precisión
        importancia = accuracy_inicial - accuracy_permutado
        importancia_variables.append(importancia)
    
    return importancia_variables


def graficar_importancia_variables(importancia_variables, nombres_variables):
    # Calculo el valor absoluto de la importancia de las variables
    importancia_absoluta = [abs(valor) for valor in importancia_variables]
    
    # Ordeno las variables y la importancia de mayor a menor
    importancia_variables_ordenadas, nombres_variables_ordenados = zip(*sorted(zip(importancia_absoluta, nombres_variables), reverse=True))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(nombres_variables_ordenados, importancia_variables_ordenadas, color='skyblue')
    ax.set_ylabel('Importancia')
    ax.set_xlabel('Variables')
    ax.set_title('Importancia de las Variables (Valor Absoluto)')
    plt.xticks(rotation=45) 
    plt.show()