# Importo las librerías principales
import pandas as pd
import functions as fun

# Importo los módulos de la libreriía SKLearn
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

# Main
if __name__ == "__main__":
    
    input_vars = ['ExternalRiskEstimate','NetFractionRevolvingBurden','AverageMInFile','MSinceOldestTradeOpen','PercentTradesWBalance','PercentInstallTrades','NumSatisfactoryTrades','NumTotalTrades','PercentTradesNeverDelq','MSinceMostRecentInqexcl7days']
    # Leo los archivos CSV
    X_train = pd.read_csv('Datasets/X_train.csv')
    X_test = pd.read_csv('Datasets/X_test.csv')
    y_train_df = pd.read_csv('Datasets/y_train.csv')
    y_test_df = pd.read_csv('Datasets/y_test.csv')

    # Convierto los DataFrames a Series
    y_train = y_train_df.iloc[:, 0]  # Selecciono la primera columna
    y_test = y_test_df.iloc[:, 0]  # Selecciono la primera columna

    # Compruebo las dimensiones de los conjuntos de entrenamiento y prueba
    print("Dimensiones del conjunto de entrenamiento (X_train, y_train):", X_train.shape, y_train.shape)
    print("Dimensiones del conjunto de prueba (X_test, y_test):", X_test.shape, y_test.shape)


    # 1. Identificación y ajuste de modelos de clasificación
    models = {
        "Linear Regression": LinearRegression(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Bagging": BaggingClassifier(n_estimators=100, max_samples=0.8),
        "K-NN": KNeighborsClassifier(),
        "Support Vector Machine (SVM)": SVC(kernel='linear', random_state=42),
        "Naive-Bayes": GaussianNB()
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        print(f"Training {name} completed.\n")

    # 2. Análisis comparativo de los modelos ajustados
    for name, model in models.items():
        print(f"Evaluating {name}:")
        if name == "Linear Regression":
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print("Mean Squared Error:", mse)
            print("R^2 Score:", r2)
            # Otras métricas de regresión según sea necesario
        else:
            y_pred = model.predict(X_test)
            print(classification_report(y_test, y_pred))
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
        print("="*50)
        
    
    # Analizo de una froma más precisa SVM con diferentes kernels y Naive Bayes
    # SVM KERNEL LINEAL
    print("Modelo SVM - Kernel Lineal")
    param = {'SVC__C': [0.01]}   # Parametro de coste
    pipe = Pipeline(steps=[ ('scaler', StandardScaler()), 
                            ('SVC',  SVC(kernel='linear', # kernel lineal
                                        probability=True, # para predecir probabilidades
                                        random_state=150))]) # replicacion
    # Cross Validation para encontrar el mejor parametro
    nFolds = 5
    Linear_SVC_gcv = GridSearchCV(estimator=pipe, # Estructura definida 
                            param_grid=param, # Parametros a buscar
                            n_jobs=-1, # Number of cores to use (parallelize)
                            scoring='accuracy', # Accuracy 
                            cv=nFolds) # Number of Folds 
    Linear_SVC_gcv.fit(X_train[input_vars], y_train) # Busco los parametros
    # Convierto a probabilidades
    Linear_SVC_fit = CalibratedClassifierCV(
                            estimator=Linear_SVC_gcv, # Estructura a usar
                            n_jobs=-1, # Number of cores to use (parallelize)
                            method='isotonic', # Metodo de calibracion
                            cv=nFolds) # Number of Folds 
    Linear_SVC_fit.fit(X_train[input_vars], y_train) # Busco los parametros
    # Ejecuto el modelo con los datos
    y_pred = Linear_SVC_fit.predict_proba(X_test)
    metricas = fun.classification_report(y_test, y_pred[:,1], 1)
    # Guardo los datos en el csv
    fun.escribir_resultados_en_csv("MetricasModelos.csv", "SVM Lineal", metricas)
    print("="*50)
    
    
    # SVM KERNEL Radial
    print("Modelo SVM - Kernel RBF")
    param = {'SVC__C': [1000], # Parametro de coste
            'SVC__gamma':[0.001]} # Inverse width parameter in the Gaussian Radial Basis kernel 
    pipe = Pipeline(steps=[ ('scaler', StandardScaler()), 
                            ('SVC',  SVC(kernel='rbf', # kernel radial
                                        probability=True, # para predecir probabilidades
                                        random_state=150))]) # replicacion
    # Cross Validation para encontrar el mejor parametro
    nFolds = 10
    SVC_fit = GridSearchCV(estimator=pipe, # Estructura definida 
                        param_grid=param, # Parametros a buscar
                        n_jobs=-1, # (parallelize)
                        scoring='accuracy', # Accuracy
                        cv=nFolds) # Number of Folds 
    SVC_fit.fit(X_train[input_vars], y_train) # Busco los parametros
    # Ejecuto el modelo con los datos
    y_pred = SVC_fit.predict_proba(X_test)
    metricas = fun.classification_report(y_test, y_pred[:,1], 1)
    # Guardo los datos en el csv
    fun.escribir_resultados_en_csv("MetricasModelos.csv", "SVM Radial", metricas)
    print("="*50)
    
    
    # Naive Bayes
    print("Modelo Naive Bayes")
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)
    y_pred = naive_bayes.predict_proba(X_test)
    metricas = fun.classification_report(y_test, y_pred[:,1], 1)
    # Guardo los datos en el csv
    fun.escribir_resultados_en_csv("MetricasModelos.csv", "NaiveBayes", metricas)
    print("="*50)