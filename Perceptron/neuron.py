import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def weight_init(num_inputs):
    """
    Funcao que inicializa os pesos e bias aleatoriamente utilizando numpy
    Parametro: num_inputs - quantidade de entradas X
    Retorna: w,b - pesos e bias da rede inicializados
    """
    # Insira seu codigo aqui (~2 linhas)
    w = np.random.random((num_inputs))
    b = -0.1
    return w, b


def activation_func(func_type, z):
    """
    Funcao que implementa as funcoes de ativacao mais comuns
    Parametros: func_type - uma string que contem a funcao de ativacao desejada
                z - vetor com os valores de entrada X multiplicado pelos pesos
    Retorna: saida da funcao de ativacao
    """
    # Seu codigo aqui (~2 linhas)
    if func_type == 'sigmoid':
        return 1 / (1 + np.exp(-z))
    elif func_type == 'tanh':
        return (2 / (1 + np.exp(-2 * z))) - 1
    elif func_type == 'relu':
        return np.maximum(0, z)
    elif func_type == 'degrau':
        return 1 * (z > 0)


def visualizeActivationFunc(z):
    z = np.arange(-5., 5., 0.2)
    func = []
    for i in range(len(z)):
        func.append(activation_func('tanh', z[i]))

    plt.plot(z, func)
    plt.xlabel('Entrada')
    plt.ylabel('Valores de Saida')
    plt.show()


def forward(w, b, X):
    """
    Funcao que implementa a etapa forward propagate do neuronio
    Parametros: w - pesos
                b - bias
                X - entradas
    """
    # Seu codigo aqui (~2 linhas)
    z = np.dot(w, X) + b
    # print('np.dot(w, X: ' + str(np.dot(w, X)))
    # z += b
    # print('b: ' + str(b))
    return activation_func('sigmoid', z)


def predict(out):
    """
    Funcao que aplica um limiar na saida
    Parametro: y - saida do neuronio
    """
    # Seu codigo aqui (~1 linha)
    return 1 * (out >= 0.5)


def perceptron(x, y, num_interaction, learning_rate):
    """
    Funcao que implementa o loop do treinamento
    Parametros: x - entrada da rede
                y - rotulos/labels
                num_interaction - quantidade de interacoes desejada para a
                        rede convergir
                learning_rate - taxa de aprendizado para calculo do erro
    """
    # Passo 1 - Inicie os pesos e bias (~1 linha)
    w, b = weight_init(x.shape[0])
    # Passo 2 - Loop por X interacoes
    for j in range(num_interaction):
        # Passo 3 -  calcule a saida do neuronio (~1 linha)
        y_pred = forward(w, b, x)
        # Passo 4 - calcule o erro entre a saida obtida e a saida desejada
        # 	nos rotulos/labels (~1 linha)
        erro = y - y_pred
        # Passo 5 - Atualize o valor dos pesos (~1 linha)
        # Dica: voce pode utilizar a funcao np.dot e a funcao transpose
        # 	de numpy
        w += np.dot(erro*learning_rate, x.T)

    # Verifique as saidas
    print('Saida obtida:', y_pred)
    print('Pesos obtidos:', w)

    # Metricas de Avaliacao
    y_pred = predict(y_pred)
    print('Matriz de Confusao:')
    print(confusion_matrix(y, y_pred))
    print('F1 Score:')
    print(classification_report(y, y_pred))
