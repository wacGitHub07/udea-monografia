# Librerías para estructuras de datos
import pandas as pd
import numpy as np

# Librerías sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

# Librerías para gráficar datos
import seaborn as sns
import matplotlib.pyplot as plt

# Métricas de evaluación del modelo
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import iqr # Interquartile range


###################################################################
#                  Funciones para graficar
###################################################################

def imprimir_dimensiones(data: pd.DataFrame):
    print(f"Numero de muestras: {data.shape[0]}, Número de columnas: {data.shape[1]}")

# Configurar estilo de gráficos
sns.set_style({'font.family':'serif', 'font.serif':['Times New Roman']})
def graficar_distribucion_histograma(data: pd.DataFrame, column: str) ->None:
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.title('Distribución de precios de arriendos')
    sns.histplot(data[column])
    plt.show()

def graficar_distribucion_boxplot(data: pd.DataFrame, column: str) -> None:
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.title('Distribución de precios de arriendos')
    sns.boxplot(data = data, y = column)
    plt.show()

def graficar_distribcion_vv(data: pd.DataFrame, columns:list, method:str="hist", fig_zise: tuple=(15,8), num_rows: int=2, num_cols:int=3) -> None:
    columns_type = columns
    fig, axs = plt.subplots(nrows=num_rows,ncols=num_cols)
    fig.set_figwidth(fig_zise[0])
    fig.set_figheight(fig_zise[1])
    col = 0
    for i in range(0, num_rows):    
        for j in range(0, 3):
            try:
                if method == "dist":
                    sns.scatterplot(data[columns_type[col]], ax=axs[i,j])
                else:
                    sns.histplot(data[columns_type[col]], ax=axs[i,j])
            except IndexError:
                continue
            col += 1


###################################################################
#                  Tratamiento de atípicos
###################################################################
def atipicos_iqr(data: pd.DataFrame, column: str) -> dict:
    # Definición del rango intercuartil
    iqr_result = iqr(data[column], axis = 0, rng = (25, 75), interpolation = 'midpoint')
    print("Rango intercuartil para variable precio: ", iqr_result)

    # Calculo de percentiles y límites inferior y superior
    q1 = np.percentile(data[column], 25, method = 'midpoint')
    q3 = np.percentile(data[column], 75, method = 'midpoint')

    min_limit = q1 - 1.5*iqr_result
    max_limit = q3 + 1.5*iqr_result

    print(f"quartil 1: {q1} quartil 3: {q3}")
    print("Límites inferiores = ", min_limit)
    print("Límites superiores = ", max_limit)

    return {'iqr': iqr_result, 'q1': q1, 'q3': q3, 'min_limit': min_limit, 'max_limit': max_limit}


###################################################################
#               Estandarizacion de datos
###################################################################
def estandarizar(data: pd.DataFrame) -> pd.DataFrame:
    MM = MinMaxScaler()
    x_norm = MM.fit_transform(data)
    x_norm = pd.DataFrame(x_norm, columns=data.columns)

    return x_norm


###################################################################
#               Importancia de variables
###################################################################
