# Importo las librerías principales
import pandas as pd
import classification_models as clsf_modl
import functions as fun
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Main
if __name__ == "__main__":
    
    input_vars = ['ExternalRiskEstimate','NetFractionRevolvingBurden','AverageMInFile','MSinceOldestTradeOpen','PercentTradesWBalance','PercentInstallTrades','NumSatisfactoryTrades','NumTotalTrades','PercentTradesNeverDelq','MSinceMostRecentInqexcl7days']
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
    
    
    # KNN
    print("KNN Model")
    # Minkowski con p=2 es la distancia Euclídea
    p = 2 
    # k optima calculada previamente
    optimal_k = 29
    # Creo el modelo
    knn_fit = clsf_modl.knn(k=optimal_k, p=p)
    print(knn_fit)
    # Entreno el modelo
    knn_fit.fit(X_train_array, y_train_array, k=optimal_k, p=p)
    # Obtengo las métricas de cómo ha funcionado en el test
    y_test_prob = knn_fit.predict_proba(np.array(X_test))
    metricas = fun.classification_report(y_test, y_test_prob[:, 1], 1)
    # Guardo los datos en el csv
    fun.escribir_resultados_en_csv("MetricasModelos.csv", "KNN", metricas)
    print("="*50)
    
    
    # Linear Regression
    print("Linear Regression Model")
    # Creo el modelo y lo entreno
    linreg = clsf_modl.LinearRegressor()
    linreg.fit(X_train, y_train)
    # Obtengo las métricas de cómo ha funcionado en el test
    y_test_prob = linreg.predict_proba(X_test)
    metricas = fun.classification_report(y_test, y_test_prob, 1)
    # Guardo los datos en el csv
    fun.escribir_resultados_en_csv("MetricasModelos.csv", "LinearRegression", metricas)
    print("="*50)
    
    
    # Logistic Regression
    print("Logistic Regression Model")
    # Aplico la regularización ElasticNet
    # Creo un array con estructura logaritmica
    pow_min = -5
    pow_max = 4  
    num_values = 11
    C_values = np.logspace(pow_min, pow_max, num=num_values) # Vector de \lambda 
    weights_evolution = []  
    accuracies = []
    # Creo un modelo con cada valor de lambda
    for C in C_values:
        model = clsf_modl.LogisticRegressor()  
        model.fit(X_train, y_train, penalty='elasticnet', C=C)
        weights_evolution.append(model.weights)
        # Calculo la accuracy
        y_pred = model.predict(X_test)
        accuracy = fun.accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    # Muestro la evolucion de los pesos
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    for feature_index in range(len(weights_evolution[0])):
        weight_trajectory = [weights[feature_index] for weights in weights_evolution]
        plt.plot(C_values, weight_trajectory, label=f'Feature {feature_index + 1}')
    plt.xscale('log')
    plt.xlabel('C (Inverse of Regularization Strength)')
    plt.ylabel('Weight Magnitude')
    plt.title('Evolution of Weights with Respect to C')
    plt.legend(loc='best')
    plt.grid(True)
    # Encuentro la accuracy máxima
    max_accuracy = max(accuracies)
    # Veo el valor de lambda que corresponde con esa accuracy
    optimum_C_values = [C for C, acc in zip(C_values, accuracies) if acc == max_accuracy]
    optimum_C = max(optimum_C_values)
    print(f"The optimum value of C based on accuracy is: {optimum_C}")
    # Muestro la accuracy en funcion de C
    plt.subplot(1, 2, 2)
    plt.plot(C_values, accuracies, marker='o', linestyle='-')
    plt.plot(optimum_C, max_accuracy, 'ro', markersize=10, label='Optimum C')
    plt.xscale('log')
    plt.xlabel('C (Inverse of Regularization Strength)')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy with Respect to C')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Creo el modelo y lo entreno con la regularización óptima
    LogReg_optimum_C_ElasticNet = clsf_modl.LogisticRegressor()
    LogReg_optimum_C_ElasticNet.fit(X_train, y_train, penalty = 'elasticnet', C = optimum_C, verbose = False)
    # Obtengo las métricas de cómo ha funcionado en el test
    y_test_prob = LogReg_optimum_C_ElasticNet.predict_proba(X_test)
    metricas = fun.classification_report(y_test, y_test_prob, 1)
    # Guardo los datos en el csv
    fun.escribir_resultados_en_csv("MetricasModelos.csv", "LogisticRegression", metricas)
    print("="*50)
    
    
    # Decision Tree
    print("Decision Tree Model")
    # Mínimo numero de observacions y altura del arbol óptimo calculado previamente
    min_num_samples = 48
    max_depth = 6
    # Creo el modelo y lo entreno
    decision_tree = clsf_modl.DecisionTree(min_samples=min_num_samples, max_depth=max_depth)
    decision_tree.fit(X_train_array, y_train_array)
    # Obtengo las métricas de cómo ha funcionado en el test
    y_test_prob = decision_tree.predict(np.array(X_test))
    metricas = fun.classification_report(y_test, np.array(y_test_prob), 1)
    # Guardo los datos en el csv
    fun.escribir_resultados_en_csv("MetricasModelos.csv", "DecisionTree", metricas)
    print("="*50)
    
    
    # Bagging
    print("Bagging Model")
    # Creo el modelo y lo entreno
    bagging_model = clsf_modl.Bagging(n_estimators=100, min_num_samples = min_num_samples, max_depth=max_depth)
    bagging_model.fit(X_train_array, y_train_array)
    # Obtengo las métricas de cómo ha funcionado en el test
    y_test_prob = bagging_model.predict(np.array(X_test))
    metricas = fun.classification_report(y_test, np.array(y_test_prob), 1)
    # Guardo los datos en el csv
    fun.escribir_resultados_en_csv("MetricasModelos.csv", "Bagging", metricas)
    print("="*50)
    
    
    # Random Forest
    print("Random Forest Model")
    # Creo el modelo y lo entreno
    random_forest = clsf_modl.RandomForest(n_estimators=100, min_num_samples = min_num_samples, max_depth=max_depth)
    random_forest.fit(X_train_array, y_train_array)
    # Obtengo las métricas de cómo ha funcionado en el test
    y_test_prob = random_forest.predict(np.array(X_test))
    metricas = fun.classification_report(y_test, np.array(y_test_prob), 1)
    # Guardo los datos en el csv
    fun.escribir_resultados_en_csv("MetricasModelos.csv", "RandomForest", metricas)
    print("="*50)