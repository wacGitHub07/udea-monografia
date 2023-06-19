# Monografía UdeA Especialización en Analítica Corte 4

## Descripción
Este repositorio contiene la solución al proyecto de grado de la Especializacion en Analítica de la Universidad de Antioquia corte 4.

## Proyecto

**Predicción de precios de arriendos en la ciudad de Medellín en base a información obtenida con Web Scraping**


## Contenido

### 1_Scraping

El proceso de scraping consiste en la extracción de información de inmuebles en arriendo desde el sitio web https://www.espaciourbano.com/. El contenido de este módulo se divide en lo siguiente
- Notebook 1_scraping_espacio_urbano: Contiene el código necesario para realizar las peticiones al sitio y procesar la respuesta con los datos deseados
- Notebook 2_scraping_datos_consolidados: Consolida la información extraida del sitio, ya que esta corresponde a la consulta de varias zonas en múltiples páginas
- utils: Contiene archivos de ayuda para dar formato y construir una base tabulada para la fase de modelado
- datos modelo: Contiene los datos utilizados en la fecha de creación del modelo. Esto debido a que al ser un sitio activo la información es constantemente cambiante

### 2_Model

El proceso de contrucción del modelo corresponde a la ejecución de múltiples iteraciones, partiendo desde la construccion de un modelo línea base hasta la implementación de estrategias mas complejas en la búsqueda de la meta del negocio la cual es: "contruir un modelo de regresión para la predicción de precios de arriendo en la ciudad de Medellín cuya metrica MAPE no debe superar el 15%. El modelo debe ser evaluado por zonas para ser utilizado solo en aquellas donde se cumpla la métrica".

El contenido de este módulo se divide en lo siguiente:
 - 1_Datos: 
    - Notebook 1_tratamiento_datos: Contiene la preparación adicional a los datos en la construcción de la base de entrenamiento.
    - base_modelado.csv: Base de entrenamiento

- 2_Modelo:
    - funciones.py: Script con funciones transversales útiles y reutilizables a lo largo de todas las iteraciones
    - iteraciones : 
        - iteracion_1: Iteracion inicial, análisis de la variable objetivo y construcción del modelo línea base
        - iteracion_2: Tratamiento general a los datos: distribución, atípicos, correlación y re ejecución de modelo línea base en busca de mejorar las métricas
        - iteración_3: Ejecución de modelos de regresión tradicionales: Regresión Lineal, Regresión Lasso, Árbol de Decisión, Bosques Aleatorios y Maquinas de Soporte Vectorial
        - iteracion_4: Ejecución de modelos de boosting: Ada Boost, Gradient Boosting, XGBoost
        - iteracion_5: Ejecución de sub iteraciones que corresponden al análisis por zonas
          - iteracion 5_1: Sub iteración eliminando zona minoritaria (San Antonio de Prado)
          - iteracion 5_2: Sub iteración eliminando (San Antonio de Prado y centro)
          - iteracion 5_3: Sub iteración eliminando (San Antonio de Prado, centro y belén)
          - iteracion 5_4: Ejecución sobre viviendas con precio inferior o igual a __6000000COP__
        - iteracion 6: Implementación de clustering y entrenamiento de modelos de regresión
        - iteracion 7: Construcción de modelo final

Nota Importantes: 
- Algunas iteraciones contienen un archivo readme para contextualizar al lector a cerca del proceso ejecutado
- Solo se consideran modelos cuya diferencia entre R2 Entrenamiento y R2 Test sea menor o igual a 0.05, si un resultado no cumple esto se determina que la ejecución no es concluyente

### Trabajo Futuro
- Eliminar la necesidad de interactuar con las páginas hasta encontrar las urls deseadas, este proceso se debe automatizar
- Realizar consulta de datos sobre otros sitios de arriendos de inmuebles para conseguir mas cobertura sobre la ciudad
- Implementación del modelo en el negocio
- Implementar proceso de scraping para ejecución periodica y almacenamiento histórico
- Contruir proceso automático de alerta de reentrenamiento y recalibración del modelo