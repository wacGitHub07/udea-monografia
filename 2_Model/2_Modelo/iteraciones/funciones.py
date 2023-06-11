# Librerías para estructuras de datos
import pandas as pd
import numpy as np

# Librerías sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Librerías para gráficar datos
import seaborn as sns
import matplotlib.pyplot as plt

# Métricas de evaluación del modelo
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import iqr # Interquartile range

# Correlación
from scipy.stats import spearmanr

# Modelos
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


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
#                  Correlación
###################################################################
def correlacion_vs_objetivo(data_pred: pd.DataFrame, data_tar: pd.DataFrame) -> None:
    spearman = []
    valor_p = []
    for col in data_pred:
        s_valor, p_valor = spearmanr(data_pred[col], data_tar)
        spearman.append(s_valor)
        valor_p.append(p_valor)

    # Cálculo de correlación absoluta
    spearman = [abs(i) for i in spearman]

    # Tamaño de la figura
    fig, ax = plt.subplots(figsize =(16, 12))
    # Horizontal Bar Plot
    ax.barh(data_pred.columns, spearman)

    # Formato de la gráfica
    ax.grid( color ='black',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.2)
    ax.invert_yaxis()
    # Añadir labels a las barras
    for i in ax.patches:
        plt.text(i.get_width()+0.03, i.get_y()+0.5,
                str(round((i.get_width()), 2)),
                fontsize = 8, fontweight ='bold',
                color ='grey')

    plt.axvline(0.2,0,1,label='correlation limit',c='red')
    # Mostrar gráfico
    plt.show()

def correlacion_tipo_variable(data_pred :pd.DataFrame, features: list, figsize: tuple=(5,5)) -> None:
    spearman_num = []
    valor_p_num = []
    data_num = data_pred[features].values
    data_num = np.asanyarray(data_num)

    for _,n in enumerate(np.arange(0,data_num.shape[1])):
        for _,m in enumerate(np.arange(0,data_num.shape[1])):  
            s_valor, p_valor = spearmanr(data_num[:,n], data_num[:,m])
            spearman_num.append(s_valor)
            valor_p_num.append(p_valor)

    spearman_num = np.asarray(spearman_num)
    spearman_r = spearman_num.reshape(data_num.shape[1],data_num.shape[1])

    plt.figure(figsize=figsize)
    ax = sns.heatmap(spearman_r, annot=True, fmt='g', xticklabels = features, yticklabels = features)
    plt.show()

def correlacion_completa(data_pred :pd.DataFrame, limit :float=0.7) -> None:
    spearman_num = []
    valor_p_num = []
    data_num = data_pred.values
    data_num = np.asanyarray(data_num)

    for _,n in enumerate(np.arange(0,data_num.shape[1])):
        for _,m in enumerate(np.arange(0,data_num.shape[1])):  
            s_valor, p_valor = spearmanr(data_num[:,n], data_num[:,m])
            spearman_num.append(s_valor)
            valor_p_num.append(p_valor)
            
    corr = 0
    columns = list(data_pred.columns)
    for i in columns:
        for j in columns:
            if spearman_num[corr] >= 0.7 and i != j:
                print("{} vs {}: {}".format(i, j, spearman_num[corr]))
            corr += 1

        

###################################################################
#                  Construcción de Modelo
###################################################################
def get_model(model:str, params:dict, seed:int) -> any:
    if model == "LinearRegression":
        return LinearRegression(**params)
    elif model == "Lasso":
        return linear_model.Lasso(**params, random_state=seed)
    elif model == "DecisionTreeRegressor":
        return DecisionTreeRegressor(**params, random_state=seed)
    elif model == "RandomForestRegressor":
        return RandomForestRegressor(n_jobs=-1,random_state=123,**params)
    elif model == "SVR":
        return SVR(**params)
    elif model == "MLPRegressor":
        return MLPRegressor(**params, random_state=seed)
    


def ejecutar_modelo(model :str, 
                    x_train:pd.DataFrame, 
                    y_train:pd.DataFrame, 
                    x_test:pd.DataFrame, 
                    y_test:pd.DataFrame, 
                    params:dict, 
                    filename: str, 
                    seed: int=123) -> None:

    resultados = {'params' : [],
                  'R2_train': [],
                  'R2_test': [],
                  'RMSE_train': [],
                  'RMSE_test': [],
                  'MAPE_train': [],
                  'MAPE_test': [],}
    
    for pars in params:

        modelo = get_model(model = model, params=pars, seed=seed)
        modelo.fit(x_train, y_train)

        resultados['params'].append(pars)
        resultados['R2_train'].append(r2_score(y_true=y_train, y_pred=modelo.predict(x_train)))
        resultados['R2_test'].append(r2_score(y_true=y_test, y_pred=modelo.predict(x_test)))
        resultados['RMSE_train'].append(mean_squared_error(y_true=y_train, y_pred=modelo.predict(x_train), squared=False))
        resultados['RMSE_test'].append(mean_squared_error(y_true=y_test, y_pred=modelo.predict(x_test), squared=False))
        resultados['MAPE_train'].append(mean_absolute_percentage_error(y_true=y_train, y_pred  = modelo.predict(x_train)))
        resultados['MAPE_test'].append(mean_absolute_percentage_error(y_true=y_test, y_pred  = modelo.predict(x_test)))
    
        print(f"Modelo: {pars}")

    resultados = pd.DataFrame(resultados)
    resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
    resultados = resultados.drop(columns = 'params')
    resultados["diff"] = abs(resultados['R2_train'] - resultados['R2_test'])
    resultados = resultados.sort_values('diff')
    resultados = resultados[resultados['diff'] <= 0.05] # Control de overfitting
    resultados.to_csv(f'{filename}.csv', index=False)