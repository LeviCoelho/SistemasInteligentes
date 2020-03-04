# SÃ³ consegue separar em duas coisas
import numpy as np # Biblioteca para algebra linear
from activation_functions import signum_function
import matplotlib.pyplot as plt # Responsavel pelos graficos=

class Adaline():
        
    def __init__(self, input_size, act_func=signum_function, epochs=100, learning_rate=0.0001, ErroPermitido = 1e-6, labelGraphic = 'Adaline'):
        self.act_func = act_func
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size + 1)
        self.p = 0
        self.Eqm = 0
        self.ErroPermitido = ErroPermitido
        self.labelGraphic = labelGraphic
        self.vetorEQmAnt = []       
        self.vetorE = []
        self.vetorU = []
        self.truePositive = 0
        self.falseNegative = 0
        self.falsePositive = 0
        self.trueNegative = 0
        self.Inputs = []
        self.posicao = 0
    def Graphic(self):   
        plt.figure(num = 1,figsize=(15, 8))
        plt.subplot(1, 2, 1)
        plt.scatter(self.vetorE, self.vetorEQmAnt, label = self.labelGraphic, color = 'r', marker = '.', s = 10)
        plt.legend()
        plt.ylabel('EQm')
        plt.xlabel('Epocas')
        plt.title('Grafico EQm: ' + self.labelGraphic)      
        
        plt.subplot(1, 2, 2)
        text = ['True Positive', 'False Negative', 'False Positive', 'True Negative']
        qty = [self.truePositive, self.falseNegative, self.falsePositive, self.trueNegative]
        plt.bar(text, qty, color = 'r')
        plt.xticks(text)
        plt.ylabel('Quantidade de resultados')
        plt.xlabel('Valores da Matriz de confusao')
        plt.title('Matriz de confusao: ' + self.labelGraphic)

    def showGraphic(self):        
        plt.tight_layout()
        plt.show()  
    
    def predict(self, inputs):
        inputs = np.append(-1, inputs)
        u = np.dot(inputs, self.weights)# np.dot ->MuktiplicaÃ§Ã£o de matrizes, multiplicaÃ§Ã£ vetorial
        return self.act_func(u)
        
    
    def testNetwork(self, training_inputs, labels):
        for inputs, label in zip(training_inputs, labels): # Tupla
            predicton = self.predict(inputs)
            if predicton == label and predicton == 1:
                self.truePositive = self.truePositive + 1
            if predicton != label and predicton == -1 :
                self.falseNegative = self.falseNegative + 1
            if predicton != label and predicton == 1 :
                self.falsePositive = self.falsePositive + 1
            if predicton == label and predicton == -1:
                self.trueNegative = self.trueNegative + 1
    
    def CaclEQm(self, training_inputs, labels):
        self.p = len(labels)
        Eqm = 0
        for inputs, label in zip(training_inputs, labels):
            inputs = np.append(-1, inputs)
            u = np.dot(inputs, self.weights)# np.dot ->MuktiplicaÃ§Ã£o de matrizes, multiplicaÃ§Ã£ vetorial            
            Eqm = Eqm +  (label - u)**2 #Eqm <- Eqm + (d[Atual] - u)²           
            
        return Eqm/self.p
        
    def train(self, training_inputs, labels):# X = training inputs, labels = D
         for e in range(self.epochs): #Quantidade de vezes que o algoritimo vai rodar
            print(f'>>> Start epoch {e + 1}')
            print(f'Actual weights {self.weights}')
            EQmAnt = self.CaclEQm(training_inputs, labels)
            print('Actual EQm: %f' %EQmAnt)        
           
            for inputs, label in zip(training_inputs, labels): # Tupla=
                predicton = self.predict(inputs)
                if predicton != label:
                    inputsWeights = np.append(-1, inputs)
                    self.weights = self.weights + self.learning_rate * (label - predicton) * inputsWeights #EquaÃ§Ã£o de ajuste de pesos
                    print(f'New weights {self.weights}')
                    break
                else:
                    print(f'Everything is OK!')
            
            EQmAtual = self.CaclEQm(training_inputs, labels)  
            print('New EQm: %f' %EQmAtual)
            print('New Erro: %f' %abs(EQmAtual - EQmAnt))
            self.vetorEQmAnt.append(EQmAtual)
            self.vetorE.append(e)  
            
            if abs(EQmAtual - EQmAnt) <= self.ErroPermitido: 
                    print('Adaline Concluido na epoca %i' % e)
                    break
            else :
                if e >= (self.epochs-1):
                    print('Dados nao podem ser separados atraves do Adaline')
            
