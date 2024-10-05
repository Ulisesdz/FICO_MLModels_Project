# Importo las librerías principales
import pandas as pd
import matplotlib.pyplot as plt

# Cargo los datos desde el archivo CSV
def cargar_datos_desde_csv(nombre_archivo):
    return pd.read_csv(nombre_archivo)

# Creo un gráfico de barras para una métrica
def grafico_barras_metrica(df, metrica, titulo):
    max_value = df[metrica].max()
    colors = ['skyblue' if valor != max_value else 'salmon' for valor in df[metrica]]
    
    plt.barh(df['Modelo'], df[metrica], color=colors, alpha=0.7)
    plt.xlabel(metrica)
    plt.ylabel('Modelo')
    plt.title(titulo)


if __name__ == "__main__":
    nombre_archivo = 'MetricasModelos.csv'
    datos = cargar_datos_desde_csv(nombre_archivo)

    # Métricas a graficar
    metricas_a_graficar = ['Accuracy', 'F1 Score', 'AUC ROC', 'Precision', 'Specificity', 'Recall']

    # Creo una imagen de subgráficos
    plt.figure(figsize=(12, 8))

    # Genero subgráficos para cada métrica
    for i, metrica in enumerate(metricas_a_graficar, start=1):
        plt.subplot(2, 3, i)  
        grafico_barras_metrica(datos, metrica, f'{metrica} por Modelo')
    plt.tight_layout()
    plt.show()