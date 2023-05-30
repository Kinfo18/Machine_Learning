#Ejemplo de un programa básico de Machine learning de clasificación con metodo de vecinos más cercanos
import numpy as np 
import sklearn
from sklearn.datasets import load_iris
#Para seleccionar una parte de entrenmaineto y otra de prueba
#clasificador
from sklearn.model_selection import train_test_split
#Para usar la función de encontrar vecinos más cercanos
from sklearn.neighbors import KNeighborsClassifier

iris=load_iris()

type(iris)

#Data => Caracteristicas, Target => etiquetas, Target_names => nombres de etiquetas
#Feacture_name => nombre de caracteristicas, DESCR => descripción de datos
#print(iris.keys())

# cada renglon es una flor, cada columna es una medición
#print(iris['data'])

#print(iris['target_names'])

x_train, x_test, y_train, y_test = train_test_split(iris['data'], iris['target'])

#print(x_train.shape)
#print(y_train.shape)

knn = KNeighborsClassifier(n_neighbors = 7) #Vamos a considerar los 7 vecinos más cercanos
#Para entrenar:
knn.fit(x_train, y_train)

#Para ver que -tan bien aprendio el algoritmo
#print(knn.score(x_test, y_test))

#introducimos 4 predicciones para que el algoritmo nos diga a que clase pertenece
clf = knn.predict([[1.2, 3.4, 5.6, 1.1]])

print(iris.target_names[clf])