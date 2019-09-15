# Description :

# Source :

# from keras.models import Sequential
# from keras.layers import Dense
# import numpy
#
# # задаем для воспроизводимости результатов
# numpy.random.seed(2)
#
# # загружаем датасет, соответствующий последним пяти годам до определение диагноза
# dataset = numpy.loadtxt("prima-indians-diabetes.csv", delimiter=",")
# # разбиваем датасет на матрицу параметров (X) и вектор целевой переменной (Y)
# X, Y = dataset[:,0:8], dataset[:,8]
#
# # создаем модели, добавляем слои один за другим
# model = Sequential()
# model.add(Dense(12, input_dim=8, activation='relu')) # входной слой требует задать input_dim
# model.add(Dense(15, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='sigmoid')) # сигмоида вместо relu для определения вероятности
#
# # компилируем модель, используем градиентный спуск adam
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
#
# # обучаем нейронную сеть
# model.fit(X, Y, epochs = 1000, batch_size=10)
#
# # оцениваем результат
# scores = model.evaluate(X, Y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy

# random seed for reproducibility
numpy.random.seed(2)

# loading load prima indians diabetes dataset, past 5 years of medical history
dataset = numpy.loadtxt("prima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables, splitting csv data
X = dataset[:,0:8]
Y = dataset[:,8]

# split X, Y into a train and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# create model, add dense layers one by one specifying activation function
model = Sequential()
model.add(Dense(15, input_dim=8, activation='relu')) # input layer requires input_dim param
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(1, activation='sigmoid')) # sigmoid instead of relu for final probability between 0 and 1

# compile the model, adam gradient descent (optimized)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# call the function to fit to the data (training the network)
model.fit(x_train, y_train, epochs = 1000, batch_size=20, validation_data=(x_test, y_test))