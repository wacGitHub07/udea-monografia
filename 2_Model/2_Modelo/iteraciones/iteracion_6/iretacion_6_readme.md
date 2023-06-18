## Descripción

Hasta ahora se ha identificado los siguientes comportamientos en la base de entrenamiento y los modelos:

- La distribución de la base precios tienen un sesgo hacia las casas de menor costo, dejando las zonas que tienen arriendos mas caros con muy poca presencia ante el modelo
- El analisis por zonas no muestra una solucion para encontrar la métrica deseada, incluso hubo deterioro en estas
- Al realizar una ejecución sobre un rango de precios específico hubo una mejora en las métricas, lo cual indica que el análisis se debe enfocar en la variable precio

Dado lo anterior y considerando el sesgo en la distribución de la variable precio se implementa una estrategia de clustering con el fin de generar un nueva variable en la base de entrenamiento que clasifique de manera optima los rangos de precios al interior de los datos:

### Clustering

Para seleccionar el rango de precios de manera óptima se implementará un algoritmo de kmeans considerando solo las variables __precio__ y __area_bruta__ (siendo esta última numérica y la que muestra mayor importancia en los analisis de correlacion). La cantidad de clusters a seleccionar será determinado calculando el coeficiente de suluetas. 

Luego de la implementación de los clusters se ejecuta de nuevo el entrenamiento de los modelos.

