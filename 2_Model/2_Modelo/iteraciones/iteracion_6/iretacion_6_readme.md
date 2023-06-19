## Descripción

Hasta ahora se ha identificado los siguientes comportamientos en la base de entrenamiento y los modelos:

- La distribución de la base precios tienen un sesgo hacia las casas de menor costo, dejando las zonas que tienen arriendos mas caros con muy poca presencia ante el modelo
- El analisis por zonas no muestra una solucion para encontrar la métrica deseada, incluso hubo deterioro en estas
- Al realizar una ejecución sobre un rango de precios específico hubo una mejora en las métricas, lo cual indica que el análisis se debe enfocar en la variable precio

Dado lo anterior y considerando el sesgo en la distribución de la variable precio se implementa una estrategia de clustering con el fin de generar un nueva variable en la base de entrenamiento que clasifique de manera optima los rangos de precios al interior de los datos:

### Clustering

Para seleccionar el rango de precios de manera óptima se implementará un algoritmo de kmeans considerando solo las variables __estrato__ y __area_bruta__ (siendo estas numéricas y las que muestra mayor importancia en los analisis de correlación). La cantidad de clusters a seleccionar será determinado calculando el coeficiente de suluetas. 

Luego de encontrar los labels de los clusters, este valor será incluido en una nueva variable de la base de entrenamiento para luego ejecutar los algoritmos que buscan predecir los precios de arriendos.


