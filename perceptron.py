# Só consegue separar em duas coisas
import numpy as np # Biblioteca para algebra linear
from activation_functions import signum_function
import matplotlib.pyplot as plt # Responsavel pelos graficos=

class Perceptron():
    
    def __init__(self, input_size, act_func=signum_function, epochs=100, learning_rate=0.01, labelGraphic = 'Perceptron'):
        self.act_func = act_func
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size + 1)
        self.labelGraphic = labelGraphic
         
        self.truePositive = 0
        self.falseNegative = 0
        self.falsePositive = 0
        self.trueNegative = 0
    
    def Graphic(self, x ,y,Label):
        plt.scatter(x, y, label = Label, color = 'r', marker = '.', s = 10)
        plt.legend()
        plt.show()
        
        
    def predict(self, inputs):
        inputs = np.append(-1, inputs)
        u = np.dot(inputs, self.weights)# np.dot ->Muktiplicação de matrizes, multiplicaçã vetorial
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
    
    def showConfusionMatrix(self):
        text = ['True Positive', 'False Negative', 'False Positive', 'True Negative']
        qty = [self.truePositive, self.falseNegative, self.falsePositive, self.trueNegative]
        plt.bar(text, qty, color = 'b')
        plt.xticks(text)
        plt.ylabel('Quantidade de resultados')
        plt.xlabel('Valores da Matriz de confusao')
        plt.title('Confusion Matrix')
        plt.show()
     
    def train(self, training_inputs, labels):
        error = True
        vetorE = []
        vetorY = []
        
        for e in range(self.epochs): #Quantidade de vezes que o algoritimo vai rodar
            
            error = False
            #print(f'>>> Start epoch {e + 1}')
            #print(f'Actual weights {self.weights}')
            for inputs, label in zip(training_inputs, labels): # Tupla
                #print(f'Input {inputs}')
                predicton = self.predict(inputs)
                
                vetorE.append(e)
                vetorY.append(predicton)
                
                
                if predicton != label:
                    #print(f'Expected {label}, got {predicton}. Start trainning!')
                    inputs = np.append(-1, inputs)
                    self.weights = self.weights + self.learning_rate * (label - predicton) * inputs #Equação de ajuste de pesos
                    #print(f'New weights {self.weights}')
                    error = True
                    break
                #else:
                    #print(f'Everything is OK!')
            
            #print('')
            if not error:
                print('Perceptron Concluido na epoca %i' % e)
                break
            else :
                if e >= (self.epochs-1):
                    print('Dados nao podem ser separados atraves do perceptron')
                    #self.Graphic(vetorE,vetorY,self.labelGraphic)
                
            
