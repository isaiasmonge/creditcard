# Predicción de Fraude en Transacciones de Tarjetas de Crédito

Este repositorio contiene un código en R que realiza un análisis de datos y construye varios modelos de clasificación para predecir transacciones fraudulentas en un conjunto de datos de tarjetas de crédito. Los modelos incluyen regresión logística, árbol de decisión, red neuronal artificial (ANN) y un modelo de aumento de gradiente (GBM).

## Requisitos

Antes de ejecutar el código, asegúrate de tener instaladas las siguientes bibliotecas de R:

- caret
- ranger
- data.table
- caTools
- pROC
- rpart
- rpart.plot
- neuralnet
- gbm

Puedes instalar estas bibliotecas usando `install.packages("nombre_de_la_biblioteca")`.

## Cómo usar

Sigue estos pasos para utilizar el código:

1. Clona este repositorio o descarga el archivo ZIP.

2. Ejecuta el código en RStudio u otro entorno de R.

3. El código realizará lo siguiente:
   - Carga el conjunto de datos de tarjetas de crédito.
   - Realiza una exploración de datos.
   - Manipula los datos, estandarizando la columna "Amount".
   - Divide los datos en conjuntos de entrenamiento y prueba.
   - Ajusta modelos de regresión logística, árbol de decisión, ANN y GBM.
   - Evalúa los modelos y calcula las curvas ROC.

4. Examina los resultados y las métricas de rendimiento de los modelos.



