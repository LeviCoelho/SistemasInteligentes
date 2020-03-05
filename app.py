import pandas as pd # Responsavel por ler arquivo de base de dados, pd se refere a pandas
import numpy as np # Biblioteca para algebra linear
from perceptron import Perceptron# importe o arquivo perceptron classe Perceptron
from adaline import Adaline

#Preparando dados Rochas
dataset_Rocha = pd.read_csv('databases/sonar.all-data') # caminho relativo do arquivo q está na mesma pasta do programa
dataset_Rocha.replace(['R', 'M'], [1, -1], inplace=True)
X_Rocha = dataset_Rocha.iloc[:, 0:60].values # o iloc gera uma subtabela ":" todas as linhas, da coluna 0 a 3, o ultimo numero não está incluso
D_Rocha = dataset_Rocha.iloc[:, 60:].values


#Preparando dados Iris
dataset_Iris = pd.read_csv('databases/iris.data') # caminho relativo do arquivo q está na mesma pasta do programa
dataset_Iris.replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [1, -1, -1], inplace=True)
X = dataset_Iris.values
d = dataset_Iris.values
x = np.arange(len(dataset_Iris))
np.random.shuffle(x)

X_new = X[x]
d_new = d[x]

X_Iris = X_new.iloc[0:101, 0:4].values # o iloc gera uma subtabela ":" todas as linhas, da coluna 0 a 3, o ultimo numero não está incluso
#np.random.shuffle(X_Iris)
D_Iris = d_new.iloc[0:101, 4:].values
#np.random.shuffle(D_Iris)

X_IrisTest = X_new.iloc[101:151, 0:4].values # o iloc gera uma subtabela ":" todas as linhas, da coluna 0 a 3, o ultimo numero não está incluso
#np.random.shuffle(X_IrisTest)
D_IrisTest = d_new.iloc[101:151:, 4:].values
#np.random.shuffle(D_IrisTest)
#Adaline
    #Adaline Iris
print('Rodando Adaline para dados - Iris')
a_Iris = Adaline(len(X_Iris[0]), epochs=1000, labelGraphic = 'Adaline - Iris') #epochs, quantidade de iteraçoes
a_Iris.train(X_Iris, D_Iris)
a_Iris.testNetwork(X_IrisTest,D_IrisTest)
a_Iris.Graphic()
a_Iris.showGraphic()
print('')
print('')

    #Adaline Rochas
print('Rodando Adaline para dados - Rochas')

a_Rocha = Adaline(len(X_Rocha[0]), epochs=1000)#epochs, quantidade de iteraçoes
a_Rocha.train(X_Rocha, D_Rocha)
a_Rocha.testNetwork(X_Rocha,D_Rocha)
a_Rocha.Graphic()
a_Rocha.showGraphic()
print('')

Perceptron
    #Perceptron Iris
print('Rodando Perceptron para dados - Iris')
p_Iris = Perceptron(len(X_Iris[0]), epochs=100, labelGraphic = 'Perceptron - Iris') #epochs, quantidade de iteraçoes
p_Iris.train(X_Iris, D_Iris)
p_Iris.testNetwork(X_IrisTest,D_IrisTest)
p_Iris.showConfusionMatrix()
print('')
    #Perceptron Rochas
#print('Rodando Perceptron para dados - Rochas')

#p_Rocha = Perceptron(len(X_Rocha[0]), epochs=100, labelGraphic = 'Perceptron - Rochas') #epochs, quantidade de iteraçoes
#p_Rocha.train(X_Rocha, D_Rocha)
#print('')
print('Finalizado')

