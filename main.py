'''
Samuel Pedro Campos Sena - EF3494
Saulo Miranda Silva - EF3475
Trabalho prático 3 - CCF480 - MetaHeurísticas
'''

'''
Referências utilizadas:

'''
import matplotlib.pyplot as plt
from numpy.random import seed, rand, randn
from numpy import sin, sqrt, asarray, mean, std
from Source.code import *

#1
Resultados_Treinamento_A = []
Resultados_Treinamento_B = []
#2
Resultados_Teste_A = []
Resultados_Teste_B = []

Melhor_Resultados_Teste_A = None

Melhor_Resultados_Teste_B = None

#fazer 30 vezes para cada funcao objetivo e algoritmo, adicionar em listas correspondentes
'''for i in range(30):'''

# adicionando seed de gerador rand
seed()


archive = open('Resultados.txt', 'w')
archive.write("\nConfiguracoes:")
print("Configuracoes:")
'''print("Iteracoes no HC: ", iteracoes)
print("Tamanho do passo no HC: ",Tam_passoHC)
print("Quantidade de reinicios no ILS: ", reinicios)
print("Tamanho da pertubação no ILS: ", Tam_P)
print("\nRealizando 30 iteracoes dos algoritmos...")'''
'''for i in range(30):
    _ , valor = ils(objetivo1, limites1a, iteracoes, Tam_passoHC, reinicios, Tam_P)
    Resultados_Treinamento_B.append(valor)
    _ , valor = melhor, valor = hc(objetivo1, limites1a, iteracoes, Tam_passoHC)   
    Resultados_Treinamento_A.append(valor)
    _ , valor = ils(objetivo1, limites1b, iteracoes, Tam_passoHC, reinicios, Tam_P)
    Resultados_Teste_B.append(valor)
    _ , valor = melhor, valor = hc(objetivo1, limites1b, iteracoes, Tam_passoHC)   
    Resultados_Teste_A.append(valor)
    _ , valor = ils(objetivo2, limites2c, iteracoes, Tam_passoHC, reinicios, Tam_P)
    Resultados2C_ILS.append(valor)
    _ , valor = melhor, valor = hc(objetivo2, limites2c, iteracoes, Tam_passoHC)   
    Resultados2C_HC.append(valor)
    _ , valor = ils(objetivo2, limites2d, iteracoes, Tam_passoHC, reinicios, Tam_P)
    Resultados2D_ILS.append(valor)
    _ , valor = melhor, valor = hc(objetivo2, limites2d, iteracoes, Tam_passoHC)   
    Resultados2D_HC.append(valor)'''

archive.write("\n----------------------")
archive.write("\nFuncao Objetivo 1 - AG  Config A:")
archive.write("\nMin: " + str(min(Resultados_Treinamento_A)))
archive.write("\nMax: " + str(max(Resultados_Treinamento_A)))
archive.write("\nMedia: " + str(mean(Resultados_Treinamento_A)))
archive.write("\nStd: " + str(std(Resultados_Treinamento_A)))
print("Funcao Objetivo 1 - AG  Config A:")
print("Min: ", min(Resultados_Treinamento_A))
print("Max: ", max(Resultados_Treinamento_A))
print("Media: ", mean(Resultados_Treinamento_A))
print("Std: ", std(Resultados_Treinamento_A))
plt.boxplot(Resultados_Treinamento_A)
plt.title("Resultado Fit Treinamento Conf A")
plt.savefig('plots/Treinamento_A.png', format='png')
#plt.show()

archive.write("\n\nFuncao Objetivo 1 - AG  Config B:")
archive.write("\nMin: " + str(min(Resultados_Treinamento_B)))
archive.write("\nMax: " + str(max(Resultados_Treinamento_B)))
archive.write("\nMedia: " + str(mean(Resultados_Treinamento_B)))
archive.write("\nStd: " + str(std(Resultados_Treinamento_B)))
archive.write("\n----------------------")
print("Funcao Objetivo 1 - AG  Config B:")
print("Min: ", min(Resultados_Treinamento_B))
print("Max: ", max(Resultados_Treinamento_B))
print("Media: ", mean(Resultados_Treinamento_B))
print("Std: ", std(Resultados_Treinamento_B))
plt.clf()
plt.boxplot(Resultados_Treinamento_B)
plt.title("Resultado Fit Treinamento Conf B")
plt.savefig('plots/Treinamento_B.png', format='png')
#plt.show()


archive.write("\nFuncao Objetivo 2 - AG  Config A:")
archive.write("\nMin: " + str(min(Resultados_Teste_A)))
archive.write("\nMax: " + str(max(Resultados_Teste_A)))
archive.write("\nMedia: " + str(mean(Resultados_Teste_A)))
archive.write("\nStd: " + str(std(Resultados_Teste_A)))
print("Funcao Objetivo 2 - AG  Config A:")
print("Min: ", min(Resultados_Teste_A))
print("Max: ", max(Resultados_Teste_A))
print("Media: ", mean(Resultados_Teste_A))
print("Std: ", std(Resultados_Teste_A))
plt.clf()
plt.boxplot(Resultados_Teste_A)
plt.title("Resultado Fit Teste Conf A")
plt.savefig('plots/Teste_A.png', format='png')
#plt.show()

archive.write("\n\nFuncao Objetivo 2 - AG  Config B:")
archive.write("\nMin: " + str(min(Resultados_Teste_B)))
archive.write("\nMax: " + str(max(Resultados_Teste_B)))
archive.write("\nMedia: " + str(mean(Resultados_Teste_B)))
archive.write("\nStd: " + str(std(Resultados_Teste_B)))
archive.write("\n----------------------")
print("ILS: ")
print("Min: ", min(Resultados_Teste_B))
print("Max: ", max(Resultados_Teste_B))
print("Media: ", mean(Resultados_Teste_B))
print("Std: ", std(Resultados_Teste_B))
plt.clf()
plt.boxplot(Resultados_Teste_B)
plt.title("Resultado Fit Teste Conf B")
plt.savefig('plots/Teste_B.png', format='png')
#plt.show()
archive.close()

'''
print(Resultados_Treinamento_A)
print(Resultados_Treinamento_B)
print(Resultados_Teste_A)
print(Resultados_Teste_B)
'''
