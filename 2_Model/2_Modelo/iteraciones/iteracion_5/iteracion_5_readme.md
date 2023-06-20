## Descripción

Esta iteración se caracteriza por subdividirse en una serie de sub iteraciones donde en cada una se procederá a ejecutar los modelos que han mostrado mejores métricas hasta ahora, iterando por los diferentes valores que tiene la variable __zona__ en la base de entrenamiento, los cuales son: Poblado, Laureles, Centro, Belén y San antonio de prado. La iteraciones sobre __zona__ consisten en ir eliminando las zonas con menor cantidad de registros y observar si los modelos obtienen mejores métricas, esto determinará si el modelo a a buscar se define por un grupo específico de datos obtenido por el proceso de scraping.

La sub iteraciones a ejecutar son:
- **iteracion_5_1:** Base con zonas Poblado, Laureles, Belen y Centro
- **iteracion_5_2:** Base con zonas Poblado, Laureles, Belen
- **iteracion_5_3:** Base con zonas Poblado, Laureles
- **iteracion_5_4:** Base con rango de precios

Dado que en cada sub iteración se está cambiando la base de entrenamiento se debe considerar repetir el tratamiento de datos de datos de distribución y correlación realizado en la iteración dos, así como una variación de los hiperparámetros si así lo necesita cada algoritmo.