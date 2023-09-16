# Cargamos las bibliotecas necesarias
library(caret)
library(ranger)
library(data.table)
library(caTools)

# Agregamos el conjunto de datos desde una ubicación local
creditcard_data <- read.csv("/Users/isaiascampana/R/Proyectos/Finanzas/creditcard/creditcard.csv")

# Data Exploration (Exploración de datos)
dim(creditcard_data) # Dimensiones del conjunto de datos
head(creditcard_data, 6) # Muestra las primeras 6 filas del conjunto de datos
tail(creditcard_data, 6) # Muestra las últimas 6 filas del conjunto de datos
table(creditcard_data$Class) # Cuenta las clases de transacciones (fraudulentas y no fraudulentas)
summary(creditcard_data$Amount) # Resumen estadístico de la columna 'Amount'
names(creditcard_data) # Muestra los nombres de las columnas
var(creditcard_data$Amount) # Calcula la varianza de la columna 'Amount'
sd(creditcard_data$Amount) # Calcula la desviación estándar de la columna 'Amount'

# Data Manipulation (Manipulación de datos)
head(creditcard_data) # Muestra las primeras filas antes de la manipulación
creditcard_data$Amount = scale(creditcard_data$Amount) # Estandariza la columna 'Amount'
NewData = creditcard_data[, -c(1)] # Crea un nuevo conjunto de datos sin la primera columna
head(NewData) # Muestra las primeras filas del nuevo conjunto de datos

# Data Modelling (Modelado de datos)
library(caTools)
set.seed(123)
data_sample = sample.split(NewData$Class, SplitRatio = 0.80) # Divide los datos en entrenamiento y prueba
train_data = subset(NewData, data_sample == TRUE) # Conjunto de entrenamiento
test_data = subset(NewData, data_sample == FALSE) # Conjunto de prueba
dim(train_data) # Dimensiones del conjunto de entrenamiento
dim(test_data) # Dimensiones del conjunto de prueba

# Fitting Logistic Regression Model (Ajuste del modelo de regresión logística)
Logistic_Model = glm(Class ~ ., test_data, family = binomial()) # Ajusta el modelo logístico
summary(Logistic_Model) # Muestra un resumen del modelo

# Visualizing summarized model through the following plots (Visualización del modelo resumido)
plot(Logistic_Model) # Grafica el modelo logístico

# ROC Curve to assess the performance of the model (Curva ROC para evaluar el rendimiento del modelo)
library(pROC)
lr.predict <- predict(Logistic_Model, test_data, probability = TRUE) # Predicciones de probabilidad
auc.gbm = roc(test_data$Class, lr.predict, plot = TRUE, col = "blue") # Calcula y grafica la curva ROC



# Cargamos las bibliotecas necesarias
library(rpart)      # Para construir árboles de decisión
library(rpart.plot) # Para visualizar árboles de decisión

# Construcción de un árbol de decisión
decisionTree_model <- rpart(Class ~ . , creditcard_data, method = 'class')
# - Class ~ . especifica que estamos construyendo un modelo de clasificación usando todas las variables predictoras
# - 'method = class' indica que estamos realizando una clasificación

# Realizamos predicciones con el árbol de decisión
predicted_val <- predict(decisionTree_model, creditcard_data, type = 'class')
# - 'type = class' indica que queremos predicciones de clase

# Obtenemos probabilidades de clasificación
probability <- predict(decisionTree_model, creditcard_data, type = 'prob')
# - 'type = prob' indica que queremos las probabilidades de clasificación

# Visualizamos el árbol de decisión
rpart.plot(decisionTree_model)
# Esto generará un gráfico que muestra la estructura del árbol

library(neuralnet)

# Creamos un modelo de red neuronal artificial (ANN)
ANN_model <- neuralnet(Class ~ ., train_data, linear.output = FALSE)
# - 'Class ~ .' especifica el modelo de clasificación usando todas las variables predictoras
# - 'linear.output = FALSE' indica que estamos realizando una clasificación binaria

# Visualizamos la estructura de la red neuronal
plot(ANN_model)

# Realizamos predicciones con la red neuronal
predANN <- compute(ANN_model, test_data)
resultANN <- predANN$net.result
resultANN <- ifelse(resultANN > 0.5, 1, 0)
# - 'compute' obtiene las predicciones
# - 'ifelse' asigna 1 si la probabilidad es mayor a 0.5, y 0 en caso contrario

# Instalamos y cargamos la biblioteca para el modelo de aumento de gradiente (GBM)
install.packages("gbm")
library(gbm, quietly = TRUE)

# Construimos un modelo GBM
system.time(
  model_gbm <- gbm(Class ~ .,
                   distribution = "bernoulli",
                   data = rbind(train_data, test_data),
                   n.trees = 500,
                   interaction.depth = 3,
                   n.minobsinnode = 100,
                   shrinkage = 0.01,
                   bag.fraction = 0.5,
                   train.fraction = nrow(train_data) / (nrow(train_data) + nrow(test_data))
  )
)
# - Se ajusta un modelo GBM con varias configuraciones

# Evaluamos el modelo GBM
gbm.iter <- gbm.perf(model_gbm, method = "test") # Obtenemos el número óptimo de árboles
model.influence <- relative.influence(model_gbm, n.trees = gbm.iter, sort. = TRUE) # Influencia de las variables
plot(model_gbm) # Visualizamos el modelo GBM
gbm_test <- predict(model_gbm, newdata = test_data, n.trees = gbm.iter) # Realizamos predicciones
gbm_auc <- roc(test_data$Class, gbm_test, plot = TRUE, col = "red") # Calculamos la curva ROC

gbm_auc
# Muestra el área bajo la curva ROC
