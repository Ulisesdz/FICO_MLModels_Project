# Importo las librerías principales
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import functions as fun

# Main
if __name__ == "__main__":
    # Parte 1.1

    # Cargo el archivo CSV
    df = pd.read_csv("FICO_Dataset.csv", delimiter=";")

    # Cuento los NaN por variable
    nan_counts = df.isna().sum()
    print("\nNúmero de NaN por variable:")
    print(nan_counts)
    time.sleep(2.5)

    # Antes de eliminar los Nans de la salida del output, voy a ver el comportamiento de los datos
    df_con_nans = df[df["RiskPerformance"].isna()]
    # Dataset sin NaNs en la columna de salida
    df_sin_nans = df[~df["RiskPerformance"].isna()]
    # Estructuras de los datos con NaNs en la salida
    print("Estructura de los datos con NaNs en la salida:")
    print(df_con_nans.info())
    # Estructuras de los datos sin NaNs en la salida
    print("\nEstructura de los datos sin NaNs en la salida:")
    print(df_sin_nans.info())
    time.sleep(2.5)

    # Histogramas para las variables numéricas con NaNs en la salida
    df_con_nans.hist(bins=20, figsize=(15, 10))
    plt.suptitle("Histogramas de las variables numéricas con NaNs en la salida")
    plt.show()
    # Histogramas para las variables numéricas sin NaNs en la salida
    df_sin_nans.hist(bins=20, figsize=(15, 10))
    plt.suptitle("Histogramas de las variables numéricas sin NaNs en la salida")
    plt.show()

    # Elimino las filas que contienen nans en la columna del output
    # ya que carece de sentido imputar valores de salida
    df = df.dropna(subset=["RiskPerformance"])

    # Elimino las filas que contienen -9
    df = df[(df != -9).all(axis=1)]

    # Imputo los valores -8 y -7 por imputación según la mediana
    for column in df.columns:
        if df[column].dtype == "int64" or df[column].dtype == "float64":
            median_value = df.loc[~df[column].isin([-8, -7]), column].median()
            df[column] = df[column].replace([-8, -7], median_value)

    # Realizo imputación en los valores faltantes de las variables de entrada
    nan_counts = df.drop("RiskPerformance", axis=1).isna().sum()

    # Imputo los valores faltantes con la mediana de cada columna
    for column, count in nan_counts.items():
        if count > 0:
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)

    # Cuento los NaN por variable
    nan_counts = df.isna().sum()
    print("\nNúmero de NaN por variable tras la imputación:")
    print(nan_counts)
    time.sleep(2.5)

    # Resumen de los datos
    print("Resumen de los datos:")
    print(df.info())
    time.sleep(2.5)

    # Estadísticas descriptivas
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    time.sleep(2.5)

    # Estandarizo los datos
    risk_performance_column = df["RiskPerformance"]
    df = df.drop(columns=["RiskPerformance"])
    for column in df.select_dtypes(include=["int64", "float64"]):
        # Calculo la media y la desviación típica de la columna
        mean = df[column].mean()
        std = df[column].std()
        # Estandarizo la columna
        df[column] = (df[column] - mean) / std
    df.insert(0, "RiskPerformance", risk_performance_column)

    # Estadísticas descriptivas
    print("\nEstadísticas descriptivas:")
    print(df.describe())

    # Histogramas para las variables numéricas
    df.hist(bins=20, figsize=(15, 10))
    plt.suptitle("Histogramas de las variables numéricas")
    plt.show()

    # Matriz de correlación
    corr = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de correlación")
    plt.show()

    # Veo la correlacion con la variable de salida
    correlation_with_output = corr["RiskPerformance"].drop("RiskPerformance").abs()

    # Creo el gráfico de barras para visualizar las correlaciones absolutas
    plt.figure(figsize=(10, 6))
    sns.barplot(x=correlation_with_output.index, y=correlation_with_output.values)
    plt.title("Correlación Absoluta de Variables con RiskPerformance")
    plt.xlabel("Variables")
    plt.ylabel("Correlación Absoluta")
    plt.xticks(rotation=45)
    plt.show()

    # Gráfico de conteo para la variable de salida
    plt.figure(figsize=(6, 4))
    sns.countplot(x="RiskPerformance", data=df)
    plt.title("Distribución de la variable de salida")
    plt.show()

    # Relaciones entre variables
    sns.pairplot(df, hue="RiskPerformance", diag_kind="kde")
    plt.show()

    # Guardo en otro archivo CSV
    df.to_csv("Datasets/FICO_Dataset_modif.csv", index=False)

    # Parte 1.2

    # Separo entre input y output
    input = np.array(df.drop(columns=["RiskPerformance"]))
    output = np.array(df["RiskPerformance"])

    # Divido los datos entre sets test y train
    X_train, X_test, y_train, y_test = fun.train_test_split(
        input, output, test_size=0.2, stratify=output, random_state=42
    )

    # Compruebo las dimensiones de los conjuntos de entrenamiento y prueba
    print(
        "Dimensiones del conjunto de entrenamiento (X_train, y_train):",
        X_train.shape,
        y_train.shape,
    )
    print(
        "Dimensiones del conjunto de prueba (X_test, y_test):",
        X_test.shape,
        y_test.shape,
    )

    # Convierto las matrices numpy en DataFrames de pandas
    X_train_df = pd.DataFrame(
        X_train,
        columns=[
            "ExternalRiskEstimate",
            "NetFractionRevolvingBurden",
            "AverageMInFile",
            "MSinceOldestTradeOpen",
            "PercentTradesWBalance",
            "PercentInstallTrades",
            "NumSatisfactoryTrades",
            "NumTotalTrades",
            "PercentTradesNeverDelq",
            "MSinceMostRecentInqexcl7days",
        ],
    )
    X_test_df = pd.DataFrame(
        X_test,
        columns=[
            "ExternalRiskEstimate",
            "NetFractionRevolvingBurden",
            "AverageMInFile",
            "MSinceOldestTradeOpen",
            "PercentTradesWBalance",
            "PercentInstallTrades",
            "NumSatisfactoryTrades",
            "NumTotalTrades",
            "PercentTradesNeverDelq",
            "MSinceMostRecentInqexcl7days",
        ],
    )
    y_train_df = pd.DataFrame(y_train, columns=["RiskPerformance"])
    y_test_df = pd.DataFrame(y_test, columns=["RiskPerformance"])

    # Guardo los DataFrames como archivos CSV
    X_train_df.to_csv("Datasets/X_train.csv", index=False)
    X_test_df.to_csv("Datasets/X_test.csv", index=False)
    y_train_df.to_csv("Datasets/y_train.csv", index=False, header=True)
    y_test_df.to_csv("Datasets/y_test.csv", index=False, header=True)
