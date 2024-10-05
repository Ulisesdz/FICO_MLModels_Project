import numpy as np 
from scipy.stats import mode
import matplotlib.pyplot as plt

def minkowski_distance(a, b, p=2):
    """
    Compute the Minkowski distance between two arrays.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.
        p (int, optional): The degree of the Minkowski distance. Defaults to 2 (Euclidean distance).

    Returns:
        float: Minkowski distance between arrays a and b.
    """
    diferencia = []
    for i in range(len(a)):
        resultado = abs(a[i] - b[i])
        diferencia.append(resultado)

    distancia = sum(diferencia)**(1/p)

    return distancia 


class knn:
    def __init__(self, k, p, X_train=None, y_train=None):
        self.k = k
        self.p = p
        self.x_train = X_train
        self.y_train = y_train

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, k: int = 5, p: int = 2):
        """
        Fit the model using X as training data and y as target values.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Target values.
            k (int, optional): Number of neighbors to use. Defaults to 5.
            p (int, optional): The degree of the Minkowski distance. Defaults to 2.
        """
        if X_train.shape[0] !=  y_train.shape[0]:
            raise ValueError(f"Los X_train y y_train datasets no tienen el mismo tamaño")
        self.x_train = X_train
        self.y_train = y_train
        for numero in [k,p]:
            if not isinstance(numero, int) or numero <= 0:
                raise ValueError(f"{numero} tiene que ser un número int")


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided data.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class labels.
        """
        prediciones = []

        for punto in X:
            distancia = self.compute_distances(punto)
            k_nearest_neighbors = self.get_k_nearest_neighbors(distancia)

            tipo_k_neighbours = []
            for neighbor in k_nearest_neighbors:
                tipo_punto = self.y_train[neighbor]
                tipo_k_neighbours.append(tipo_punto)

            etiquetas_predd = self.most_common_label(tipo_k_neighbours)
            prediciones.append(etiquetas_predd)

        return np.array(prediciones)

    def predict_proba(self, X):
        """
        Predict the class probabilities for the provided data.

        Each class probability is the amount of each label from the k nearest neighbors
        divided by k.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class probabilities.
        """

        predicciones_proba = []

        for punto in X:
            distancias = self.compute_distances(punto)
            k_nearest_neighbors = self.get_k_nearest_neighbors(distancias)

            tipo_k_neighbours = []
            for neighbor in k_nearest_neighbors:
                tipo_punto = self.y_train[neighbor]
                tipo_k_neighbours.append(tipo_punto)

            numeros = np.bincount(tipo_k_neighbours)
            
            if 0 in tipo_k_neighbours:
                probabilidad_uno =  numeros[0] / len(tipo_k_neighbours)
            else: 
                probabilidad_uno = 0

            if 1 in tipo_k_neighbours:
                probabilidad_dos =  numeros[1] / len(tipo_k_neighbours)
            else: 
                probabilidad_dos = 0

            predicciones_proba.append([probabilidad_uno,probabilidad_dos])

        return np.array(predicciones_proba)

    def compute_distances(self, point: np.ndarray) -> np.ndarray:
        """Compute distance from a point to every point in the training dataset

        Args:
            point (np.ndarray): data sample.

        Returns:
            np.ndarray: distance from point to each point in the training dataset.
        """
        distancias = []
        for data_point in self.x_train:
            
            if not np.array_equal(data_point, point):
                distancia = minkowski_distance(data_point, point, p=2)
                if distancia is None:
                    print(f"La distancia es None para data_point: {data_point}, point: {point}")
                distancias.append(distancia)

        return distancias
                

    def get_k_nearest_neighbors(self, distances: np.ndarray) -> np.ndarray:
        """Get the k nearest neighbors indices given the distances matrix from a point.

        Args:
            distances (np.ndarray): distances matrix from a point whose neighbors want to be identified.

        Returns:
            np.ndarray: row indices from the k nearest neighbors.
        """
        
        if self.k <= 0 or self.k > len(distances):
            raise ValueError("Invalid value of k")

        indices = np.argsort(distances)[:self.k]

        return indices
            

    def most_common_label(self, knn_labels: np.ndarray) -> int:
        """Obtain the most common label from the labels of the k nearest neighbors

        Args:
            knn_labels (np.ndarray): labels from the k nearest neighbors

        Returns:
            int: most common label
        """
        etiquetas, counts = np.unique(knn_labels, return_counts=True)   
        etiqueta_mas_repetida = etiquetas[np.argmax(counts)]   

        return etiqueta_mas_repetida

    def __str__(self):
        """
        String representation of the kNN model.
        """
        return f"kNN model (k={self.k}, p={self.p})"
    
    

class LogisticRegressor:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initializes the Logistic Regressor model.

        Attributes:
        - weights (np.ndarray): A placeholder for the weights of the model. 
                                These will be initialized in the training phase.
        - bias (float): A placeholder for the bias of the model. 
                        This will also be initialized in the training phase.
        - learning_rate (float) : The step size at each iteration while moving toward a minimum of the 
                            loss function.
        - num_iterations (int) : Numero de iteraciones del algoritmo gd
        """
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
    
    
    def fit(self, X, y, penalty = None, l1_ratio = 0.5, C = 1.0, verbose = False, print_every = 100):
        """
        Fits the logistic regression model to the data using gradient descent. 
        
        This method initializes the model's weights and bias, then iteratively updates these parameters by 
        moving in the direction of the negative gradient of the loss function (computed using the 
        log_likelihood method).

        The regularization terms are added to the gradient of the loss function as follows:

        - No regularization: The standard gradient descent updates are applied without any modification.

        - L1 (Lasso) regularization: Adds a term to the gradient that penalizes the absolute value of 
            the weights, encouraging sparsity. The update rule for weight w_j is adjusted as follows:
            dw_j += (C / m) * sign(w_j) - Make sure you understand this!

        - L2 (Ridge) regularization: Adds a term to the gradient that penalizes the square of the weights, 
            discouraging large weights. The update rule for weight w_j is:
            dw_j += (C / m) * w_j       - Make sure you understand this!
            

        - ElasticNet regularization: Combines L1 and L2 penalties. 
            The update rule incorporates both the sign and the magnitude of the weights:
            dw_j += l1_ratio * gradient_of_lasso + (1 - l1_ratio) * gradient_of_ridge


        Parameters:
        - X (np.ndarray): The input features, with shape (m, n), where m is the number of examples and n is
                            the number of features.
        - y (np.ndarray): The true labels of the data, with shape (m,).
        - penalty (str): Type of regularization (None, 'lasso', 'ridge', 'elasticnet'). Default is None.
        - l1_ratio (float): The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. 
                            l1_ratio=0 corresponds to L2 penalty, 
                            l1_ratio=1 to L1. Only used if penalty='elasticnet'. 
                            Default is 0.5.
        - C (float): Inverse of regularization strength; must be a positive float. 
                            Smaller values specify stronger regularization.
        - verbose (bool): Print loss every print_every iterations.
        - print_every (int): Period of number of iterations to show the loss.



        Updates:
        - self.weights: The weights of the model after training.
        - self.bias: The bias of the model after training.
        """
        # Obtain m (number of examples) and n (number of features)
        m, n = X.shape
        
        # Initialize all parameters to 0        
        self.weights = np.zeros(n) 
        self.bias = 0
        # Execute the iterative gradient descent
        for i in range(self.num_iterations):                     
            
            # Forward propagation
            y_hat = self.predict_proba(X)
            # Compute loss
            loss = self.log_likelihood(y, y_hat)

            # Logging
            if i % print_every == 0 and verbose:
                print(f"Iteration {i}: Loss {loss}")

            # Multiplico las dos por -(1 / m) para normalizar el gradiente y hacer que este sea consistente
            # en base al tamaño del conjunto de datos y que el tamaño no afecte.
            dw = (1 / m) * np.dot(X.T, (y_hat - y)) # Derivative w.r.t. the coefficients
            db = (1 / m) * np.sum(y_hat - y) # Derivative w.r.t. the intercept

            # Regularization: 
            # Apply regularization if it is selected.
            # We feed the regularization method the needed values, where "dw" is the derivative for the 
            # coefficients, "m" is the number of examples and "C" is the regularization hyperparameter.
            if penalty == 'lasso':
                dw = self.lasso_regularization(dw, m, C)
            elif penalty == 'ridge':
                dw = self.ridge_regularization(dw, m, C)
            elif penalty == 'elasticnet':
                dw = self.elasticnet_regularization(dw, m, C, l1_ratio)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        """
        Predicts probability estimates for all classes for each sample X.

        Parameters:
        - X (np.ndarray): The input features, with shape (m, n), where m is the number of samples and 
            n is the number of features.

        Returns:
        - A numpy array of shape (m, 1) containing the probability of the positive class for each sample.
        """
        
        # z is the value of the logits. Write it here (use self.weights and self.bias):
        z = np.dot(X, self.weights) + self.bias
        
        # Return the associated probabilities via the sigmoid trasnformation (symmetric choice)
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        Predicts class labels for samples in X.

        Parameters:
        - X (np.ndarray): The input features, with shape (m, n), where m is the number of samples and n 
                            is the number of features.
        - threshold (float): Threshold used to convert probabilities into binary class labels. 
                             Defaults to 0.5.

        Returns:
        - A numpy array of shape (m,) containing the class label (0 or 1) for each sample.
        """
        
        # Predict the class for each input data given the threshold in the argument
        probabilities = self.predict_proba(X)
        classification_result = np.where(probabilities > threshold, 1, 0)
        
        return classification_result
    
    def lasso_regularization(self, dw, m, C):
        """
        Applies L1 regularization (Lasso) to the gradient during the weight update step in gradient descent. 
        L1 regularization encourages sparsity in the model weights, potentially setting some weights to zero, 
        which can serve as a form of feature selection.

        The L1 regularization term is added directly to the gradient of the loss function with respect to 
        the weights. This term is proportional to the sign of each weight, scaled by the regularization 
        strength (C) and inversely proportional to the number of samples (m).

        Parameters:
        - dw (np.ndarray): The gradient of the loss function with respect to the weights, before regularization.
        - m (int): The number of samples in the dataset.
        - C (float): Inverse of regularization strength; must be a positive float. 
                    Smaller values specify stronger regularization.

        Returns:
        - np.ndarray: The adjusted gradient of the loss function with respect to the weights, 
                      after applying L1 regularization.
        """
        
        lasso_gradient = (C / m) * np.sign(self.weights)
        return dw + lasso_gradient
    
    
    def ridge_regularization(self, dw, m, C):
        """
        Applies L2 regularization (Ridge) to the gradient during the weight update step in gradient descent. 
        L2 regularization penalizes the square of the weights, which discourages large weights and helps to 
        prevent overfitting by promoting smaller and more distributed weight values.

        The L2 regularization term is added to the gradient of the loss function with respect to the weights
        as a term proportional to each weight, scaled by the regularization strength (C) and inversely 
        proportional to the number of samples (m).

        Parameters:
        - dw (np.ndarray): The gradient of the loss function with respect to the weights, before regularization.
        - m (int): The number of samples in the dataset.
        - C (float): Inverse of regularization strength; must be a positive float. 
                     Smaller values specify stronger regularization.

        Returns:
        - np.ndarray: The adjusted gradient of the loss function with respect to the weights, 
                        after applying L2 regularization.
        """
        
        ridge_gradient = (C / m) * self.weights
        return dw + ridge_gradient

    def elasticnet_regularization(self, dw, m, C, l1_ratio):
        """
        Applies Elastic Net regularization to the gradient during the weight update step in gradient descent. 
        Elastic Net combines L1 and L2 regularization, incorporating both the sparsity-inducing properties 
        of L1 and the weight shrinkage effect of L2. This can lead to a model that is robust to various types 
        of data and prevents overfitting.

        The regularization term combines the L1 and L2 terms, scaled by the regularization strength (C) and 
        the mix ratio (l1_ratio) between L1 and L2 regularization. The term is inversely proportional to the 
        number of samples (m).

        Parameters:
        - dw (np.ndarray): The gradient of the loss function with respect to the weights, before regularization.
        - m (int): The number of samples in the dataset.
        - C (float): Inverse of regularization strength; must be a positive float. 
                     Smaller values specify stronger regularization.
        - l1_ratio (float): The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds
                            to L2 penalty, l1_ratio=1 to L1. Only used if penalty='elasticnet'. 
                            Default is 0.5.

        Returns:
        - np.ndarray: The adjusted gradient of the loss function with respect to the weights, 
                      after applying Elastic Net regularization.
        """
        lasso_gradient = self.lasso_regularization(dw, m, C)
        ridge_gradient = self.ridge_regularization(dw, m, C)
        elasticnet_gradient = l1_ratio * lasso_gradient + (1 - l1_ratio) * ridge_gradient
        return dw + elasticnet_gradient

    @staticmethod
    def log_likelihood(y, y_hat):
        """
        Computes the Log-Likelihood loss for logistic regression, which is equivalent to
        computing the cross-entropy loss between the true labels and predicted probabilities. 
        This loss function is used to measure how well the model predicts the actual class 
        labels. The formula for the loss is:

        L(y, y_hat) = -(1/m) * sum(y * log(y_hat) + (1 - y) * log(1 - y_hat))

        where:
        - L(y, y_hat) is the loss function,
        - m is the number of observations,
        - y is the actual label of the observation,
        - y_hat is the predicted probability that the observation is of the positive class,
        - log is the natural logarithm.

        Parameters:
        - y (np.ndarray): The true labels of the data. Should be a 1D array of binary values (0 or 1).
        - y_hat (np.ndarray): The predicted probabilities of the data belonging to the positive class (1). 
                            Should be a 1D array with values between 0 and 1.

        Returns:
        - The computed loss value as a scalar.
        """
        
        # Loss function (log-likelihood)
        m = y.shape[0] # Number of examples
        loss = -(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss

    @staticmethod
    def sigmoid(z):
        """x
        Computes the sigmoid of z, a scalar or numpy array of any size. The sigmoid function is used as the 
        activation function in logistic regression, mapping any real-valued number into the range (0, 1), 
        which can be interpreted as a probability. It is defined as 1 / (1 + exp(-z)), where exp(-z) 
        is the exponential of the negative of z.

        Parameters:
        - z (float or np.ndarray): Input value or array for which to compute the sigmoid function.

        Returns:
        - The sigmoid of z.
        """
        
        # Sigmoid function to convert the logits into probabilities
        sigmoid_value = 1 / (1 + np.exp(-z))
        
        return sigmoid_value
    


class DecisionTree:
    def __init__(self, max_depth=None, min_samples=None):
        """
        Inicializa el modelo de árbol de decisión.

        Argumentos:
        - max_depth (int): Profundidad máxima del árbol. Si no se especifica, se usará la cantidad de características.
        - min_samples (int): Numero de muestras minimas para cada nodo
        """
        self.max_depth = max_depth
        self.min_samples = min_samples
        
    def gini_impurity(self, labels):
        """
        Calcula la impureza de Gini de un conjunto de etiquetas.

        Argumentos:
        - labels (np.ndarray): Etiquetas del conjunto de datos.

        Devuelve:
        - float: El valor de la impureza de Gini.
        """
        # Calculo las probabilidades de cada clase
        unique_classes, class_counts = np.unique(labels, return_counts=True)
        class_probabilities = class_counts / len(labels)
        # Calculo la impureza de Gini
        gini = 1 - np.sum(class_probabilities**2)
        return gini
    
    def split_data(self, X, y, feature_index, threshold):
        """
        Divide los datos en dos conjuntos según el valor de una característica y un umbral dado.

        Argumentos:
        - X (np.ndarray): Datos de entrada.
        - y (np.ndarray): Etiquetas de los datos.
        - feature_index (int): Índice de la característica para dividir.
        - threshold (float): Umbral para la división.

        Devuelve:
        - np.ndarray, np.ndarray, np.ndarray, np.ndarray: Datos y etiquetas divididos.
        """
        # Obtengo los índices de los datos que cumplen la condición de división
        left_indices = np.where(X[:, feature_index] <= threshold)[0]
        right_indices = np.where(X[:, feature_index] > threshold)[0]
        # Divido los datos y etiquetas según los índices obtenidos
        left_X, left_y = X[left_indices], y[left_indices]
        right_X, right_y = X[right_indices], y[right_indices]
        return left_X, left_y, right_X, right_y
        
    def find_best_split(self, X, y):
        """
        Encuentra la mejor división para un conjunto de datos dado.

        Argumentos:
        - X (np.ndarray): Datos de entrada.
        - y (np.ndarray): Etiquetas de los datos.

        Devuelve:
        - int, float: Índice de la característica y umbral para la mejor división.
        """
        best_gini = float('inf')
        best_feature_index = None
        best_threshold = None
        # Itero sobre todas las características y umbrales posibles para encontrar la mejor división
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_X, left_y, right_X, right_y = self.split_data(X, y, feature_index, threshold)
                total_samples = len(left_y) + len(right_y)
                # Calculo la impureza de Gini de la división
                gini = (len(left_y) / total_samples) * self.gini_impurity(left_y) + \
                       (len(right_y) / total_samples) * self.gini_impurity(right_y)
                # Actualizo la mejor división si la impureza de Gini es menor
                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold
        return best_feature_index, best_threshold
    
    def fit(self, X, y):
        """
        Ajusta el árbol a los datos de entrenamiento.

        Argumentos:
        - X (np.ndarray): Datos de entrada.
        - y (np.ndarray): Etiquetas de los datos.
        """
        # Si no se especifica la profundidad máxima, se usa la cantidad de características
        if self.max_depth is None:
            self.max_depth = X.shape[1]
        # Creo el árbol
        self.tree = self._grow_tree(X, y, depth=0)
        
    def _grow_tree(self, X, y, depth):
        """
        Método auxiliar para construir el árbol recursivamente.

        Argumentos:
        - X (np.ndarray): Datos de entrada.
        - y (np.ndarray): Etiquetas de los datos.
        - depth (int): Profundidad actual del árbol.

        Devuelve:
        - dict: Nodo del árbol.
        """
        # Compruebo las condiciones de parada para detener la construcción del árbol
        if depth == self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples:
            return np.mean(y)
        # Encuentro la mejor división para los datos actuales
        best_feature_index, best_threshold = self.find_best_split(X, y)
        # Divido los datos y etiquetas según la mejor división encontrada
        left_X, left_y, right_X, right_y = self.split_data(X, y, best_feature_index, best_threshold)
        # Creo el nodo del árbol
        node = {'feature_index': best_feature_index, 'threshold': best_threshold}
        # Construyo recursivamente los sub-árboles izquierdo y derecho
        node['left'] = self._grow_tree(left_X, left_y, depth + 1)
        node['right'] = self._grow_tree(right_X, right_y, depth + 1)
        return node
    
    def predict(self, X):
        """
        Predice las etiquetas de clase para nuevos datos.

        Argumentos:
        - X (np.ndarray): Nuevos datos de entrada.

        Devuelve:
        - np.ndarray: Predicciones de clase.
        """
        # Realizo la predicción para cada dato en X
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def predict_proba(self, X):
        """
        Predice las probabilidades de pertenencia a cada clase para nuevos datos.

        Argumentos:
        - X (np.ndarray): Nuevos datos de entrada.

        Devuelve:
        - np.ndarray: Probabilidades de pertenencia a cada clase.
        """
        # Realizo la predicción de probabilidades para cada dato en X
        return np.array([[1 - p, p] for p in self._traverse_tree(X, self.tree)])
    
    def _traverse_tree(self, x, node):
        """
        Método auxiliar para realizar la predicción recursivamente.

        Argumentos:
        - x (np.ndarray): Datos de entrada.
        - node (dict): Nodo del árbol actual.

        Devuelve:
        - int: Predicción de clase.
        """
        # Si llego a una hoja, devuelvo su valor como predicción
        if isinstance(node, float):
            return node
        # Compruebo si debemos ir al sub-árbol izquierdo o derecho
        if x[node['feature_index']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])

    
class Bagging:
    def __init__(self, n_estimators=100, max_samples=0.8, max_depth=None, min_num_samples=None):
        """
        Inicializa el modelo de Bagging.

        Argumentos:
        - n_estimators (int): Número de clasificadores en el ensemble.
        - max_samples (float): Proporción de muestras a ser usadas para cada muestra bootstrap.
        - min_num_samples (int): Numero de muestras minimas para cada nodo
        - max_depth (int): Profundidad máxima de cada árbol. Si no se especifica, se usará la cantidad de características.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_num_samples = min_num_samples
        self.estimators = []

    def fit(self, X_train, y_train):
        """
        Ajusta el modelo Bagging a los datos de entrenamiento.

        Argumentos:
        - X_train (np.ndarray): Datos de entrenamiento.
        - y_train (np.ndarray): Etiquetas de entrenamiento.
        """
        if self.max_depth is None:
            self.max_depth = X_train.shape[1]
        n_samples = X_train.shape[0]
        n_features = X_train.shape[1]
        # Creo tantos árboles de decision como estimadores
        for _ in range(self.n_estimators):
            # Creo una muestra bootstrap
            sample_indices = np.random.choice(range(n_samples), size=int(n_samples * self.max_samples), replace=True)
            X_bootstrap = X_train[sample_indices]
            y_bootstrap = y_train[sample_indices]
            # Entreno un arbol de decision con la muestra boostrap
            decision_tree = DecisionTree(max_depth=self.max_depth, min_samples=self.min_num_samples)
            decision_tree.fit(X_bootstrap, y_bootstrap)
            # Añado el clasificador entrenado a la lista de estimadores
            self.estimators.append(decision_tree)

    def predict(self, X_test):
        """
        Predice las etiquetas de clase para nuevos datos.

        Argumentos:
        - X_test (np.ndarray): Nuevos datos de entrada.

        Devuelve:
        - np.ndarray: Predicciones de clase.
        """
        predictions = []
        for estimator in self.estimators:
            y_pred = estimator.predict(X_test)
            predictions.append(y_pred)

        # Combino las predicciones usando votación mayoritaria
        majority_vote = mode(predictions, axis=0)[0]
        return majority_vote
    
    
    
class RandomForest:
    def __init__(self, n_estimators=100, max_samples=0.8, max_depth=None, min_num_samples=None):
        """
        Inicializa el modelo de Random Forest.

        Argumentos:
        - n_estimators (int): Número de árboles en el bosque.
        - min_num_samples (int): Numero de muestras minimas para cada nodo
        - max_depth (int): Profundidad máxima de cada árbol. Si no se especifica, se usará la cantidad de características.
        - max_features (float) : Proporción de variables a considerar para cada partición
        """
        self.n_estimators = n_estimators
        self.min_num_samples = min_num_samples
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.estimators = []
        
    def fit(self, X, y):
        """
        Ajusta el modelo Random Forest a los datos de entrenamiento.

        Argumentos:
        - X (np.ndarray): Datos de entrada.
        - y (np.ndarray): Etiquetas de los datos.
        """
        if self.max_depth is None:
            self.max_depth = X.shape[1]
        n_samples = X.shape[0]
        n_features = X.shape[1]
        # Escojo raiz de p como el numero de indices a elegir
        n_selected_features = np.sqrt(n_features)

        for _ in range(self.n_estimators):
            # Creo una muestra bootstrap
            sample_indices = np.random.choice(range(n_samples), size=int(n_samples * self.max_samples), replace=True)
            X_bootstrap = X[sample_indices]
            y_bootstrap = y[sample_indices]

            # Selecciono características aleatorias
            selected_features = np.random.choice(range(n_features), size=int(n_selected_features), replace=False)
            X_bootstrap_selected = X_bootstrap[:, selected_features]

            # Entreno el modelo
            decision_tree = DecisionTree(max_depth=self.max_depth, min_samples=self.min_num_samples)
            decision_tree.fit(X_bootstrap_selected, y_bootstrap)

            # Agregar el árbol entrenado a la lista de estimadores
            self.estimators.append((decision_tree, selected_features))
            
    def predict(self, X):
        """
        Predice las etiquetas de clase para nuevos datos.

        Argumentos:
        - X (np.ndarray): Nuevos datos de entrada.

        Devuelve:
        - np.ndarray: Predicciones de clase.
        """
        predictions = []
        for decision_tree, selected_features in self.estimators:
            X_test_selected = X[:, selected_features]
            y_pred = decision_tree.predict(X_test_selected)
            predictions.append(y_pred)

        # Combino las predicciones usando votación mayoritaria
        majority_vote = mode(predictions, axis=0)[0]
        return majority_vote



class PCA:
    def __init__(self, n_components):
        """
        Inicializa la instancia de PCA.
        
        Argumentos:
        - n_components: int, número de componentes principales a retener.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio_ = None
        
    def fit(self, X):
        """
        Ajusta el modelo PCA a mis datos de entrenamiento.
        
        Argumentos:
        - X: array-like, shape (n_samples, n_features), mis datos de entrenamiento.
        """
        # Calculo la media de cada característica
        self.mean = np.mean(X, axis=0)
        # Centro mis datos
        X = X - self.mean
        # Calculo la matriz de covarianza
        cov_matrix = np.cov(X.T)
        # Calculo los autovalores y autovectores
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # Ordeno los vectores propios en función de los autovalores
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[idxs]
        eigenvalues = eigenvalues[idxs]
        # Selecciono los primeros n_components vectores propios
        self.components = eigenvectors[0:self.n_components]
        # Calculo la varianza explicada
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues / total_variance
        
    def transform(self, X):
        """
        Proyecta los datos en el nuevo espacio definido por los componentes principales.
        
        Argumentos:
        - X: array-like, shape (n_samples, n_features), los datos que quiero transformar.
        
        Retorna:
        - array-like, shape (n_samples, n_components), mis datos transformados.
        """
        # Centro los datos con respecto a la media
        X = X - self.mean
        # Proyecto los datos en el nuevo espacio
        return np.dot(X, self.components.T)