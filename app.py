import pandas as pd # Responsavel por ler arquivo de base de dados, pd se refere a pandas
import numpy as np # Biblioteca para algebra linear
from perceptron import Perceptron# importe o arquivo perceptron classe Perceptron
from adaline import Adaline

#Preparando dados Rochas
dataset_Rochas = pd.read_csv('databases/sonar.all-data') #Caminho relativo do arquivo q está na mesma pasta do programa
dataset_Rochas.replace(['R', 'M'], [1, -1], inplace=True)
X = dataset_Rochas.values #O iloc gera uma subtabela ":" todas as linhas, da coluna 0 a 3, o ultimo numero não está incluso
D = dataset_Rochas.values 
x = np.arange(len(dataset_Rochas))
np.random.shuffle(x) #embaralha as linhas da matriz

X_Rochas_new = X[x]
D_Rochas_new = D[x]

lenTrain_Rochas = int(0.8*len(dataset_Rochas)) #Define 80% dos dados para o treinamento e 20% para teste

X_Rochas = X_Rochas_new[0:lenTrain_Rochas,0:60]
D_Rochas = D_Rochas_new[0:lenTrain_Rochas,60:]

X_Rocha_teste = X_Rochas_new[lenTrain_Rochas:,0:60]
D_Rocha_teste = D_Rochas_new[lenTrain_Rochas:,60:]

#Preparando dados Iris
dataset_Iris = pd.read_csv('databases/iris.data') #Caminho relativo do arquivo q está na mesma pasta do programa
dataset_Iris.replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [1, -1, -1], inplace=True)
X = dataset_Iris.values
D = dataset_Iris.values
x = np.arange(len(dataset_Iris))
np.random.shuffle(x) #embaralha as linhas da matriz

X_new = X[x]
D_new = D[x]

lenTrain = int(0.8*len(dataset_Iris)) #Define 80% dos dados para o treinamento e 20% para teste

X_Iris = X_new[0:lenTrain,0:4]
D_Iris = D_new[0:lenTrain,4:]

X_IrisTest = X_new[lenTrain:,0:4]
D_IrisTest = D_new[lenTrain:,4:]

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

a_Rocha = Adaline(len(X_Rochas[0]), epochs=1000, labelGraphic = 'Adaline - Rochas')#epochs, quantidade de iteraçoes
a_Rocha.train(X_Rochas, D_Rochas)
a_Rocha.testNetwork(X_Rocha_teste,D_Rocha_teste)
a_Rocha.Graphic()
a_Rocha.showGraphic() 
print('')

#Perceptron
#    #Perceptron Iris
#print('Rodando Perceptron para dados - Iris')
#p_Iris = Perceptron(len(X_Iris[0]), epochs=100, labelGraphic = 'Perceptron - Iris') #epochs, quantidade de iteraçoes
#p_Iris.train(X_Iris, D_Iris)
#p_Iris.testNetwork(X_IrisTest,D_IrisTest)
#p_Iris.showConfusionMatrix()
#print('')
    #Perceptron Rochas
#print('Rodando Perceptron para dados - Rochas')

#p_Rocha = Perceptron(len(X_Rocha[0]), epochs=100, labelGraphic = 'Perceptron - Rochas') #epochs, quantidade de iteraçoes
#p_Rocha.train(X_Rocha, D_Rocha)
#print('')
print('Finalizado')

