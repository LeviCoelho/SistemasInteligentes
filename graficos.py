import matplotlib.pyplot as plt # Responsavel pelos graficos

class Graficos():
    
    def __init__(self):
        self.X = []
        self.Y = []
        self.posicaoAtualX = 0
        self.posicaoAtualY = 0
    
    
    def show(self):
        plt.show()
        
    def criarVetorX(self, x):
        self.X[self.posicaoAtualX] = x
        self.posicaoAtualX = self.posicaoAtualX + 1
        
    def criarVetorY(self, y):
        self.Y[self.posicaoAtualY] = y
        self.posicaoAtualY = self.posicaoAtualY + 1
        
        
    
    def criarGrafico(self):
        # Criando um gráfico
        plt.scatter(self.X, self.Y, label = 'u', color = 'b', marker = '*', s = 100)
        plt.legend()
    