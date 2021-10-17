'''
Samuel Pedro Campos Sena - EF3494
Saulo Miranda Silva - EF3475
Trabalho prático 3 - CCF480 - MetaHeurísticas
'''

'''
Referências utilizadas:

'''
import matplotlib.pyplot as plt
from numpy import mean, std
from Source.code import *

#1
Resultados_Treinamento_A = []
Resultados_Treinamento_B = []
hof_A = []

#2
Resultados_Teste_A = []
Resultados_Teste_B = []
hof_B = []

Melhor_Resultados_Teste_A = None

Melhor_Resultados_Teste_B = None



print("Configuracoes:")


print("Executando 30 iteracoes...")
#fazer 30 vezes para cada funcao objetivo e algoritmo, adicionar em listas correspondentes
for i in range(30):
    _, _, hof = main()
    hof = hof[0]
    func = toolbox.compile(expr=hof)
    result_test = sum(bool(func(*customer[:19])) is bool(customer[19]) for customer in test) / len(test)
    result_train = sum(bool(func(*customer[:19])) is bool(customer[19]) for customer in train) / len(train)
    Resultados_Treinamento_A.append(result_train)
    Resultados_Teste_A.append(result_test)
    hof_A.append(hof)

    _, _, hof = main(conf=False)
    hof = hof[0]
    func = toolbox.compile(expr=hof)
    result_test = sum(bool(func(*customer[:19])) is bool(customer[19]) for customer in test) / len(test)
    result_train = sum(bool(func(*customer[:19])) is bool(customer[19]) for customer in train) / len(train)
    Resultados_Treinamento_B.append(result_train)
    Resultados_Teste_B.append(result_test)
    hof_B.append(hof)

Melhor_Resultados_Teste_A = hof_A[Resultados_Teste_A.index(max(Resultados_Teste_A))]

Melhor_Resultados_Teste_B = hof_B[Resultados_Teste_B.index(max(Resultados_Teste_B))]
archive = open('Resultados.txt', 'w')
archive.write("\nConfiguracoes:")
archive.write("\n----------------------")
archive.write("\nResultados Treinamento - Config A:")
archive.write("\nMin: " + str(min(Resultados_Treinamento_A)))
archive.write("\nMax: " + str(max(Resultados_Treinamento_A)))
archive.write("\nMedia: " + str(mean(Resultados_Treinamento_A)))
archive.write("\nStd: " + str(std(Resultados_Treinamento_A)))
print("Resultados Treinamento - Config A:")
print("Min: ", min(Resultados_Treinamento_A))
print("Max: ", max(Resultados_Treinamento_A))
print("Media: ", mean(Resultados_Treinamento_A))
print("Std: ", std(Resultados_Treinamento_A))
plt.boxplot(Resultados_Treinamento_A)
plt.title("Resultados Treinamento - Config A")
plt.savefig('Plots/Treinamento_A.png', format='png')
#plt.show()

archive.write("\n\nResultados Treinamento - Config B:")
archive.write("\nMin: " + str(min(Resultados_Treinamento_B)))
archive.write("\nMax: " + str(max(Resultados_Treinamento_B)))
archive.write("\nMedia: " + str(mean(Resultados_Treinamento_B)))
archive.write("\nStd: " + str(std(Resultados_Treinamento_B)))
archive.write("\n----------------------")
print("Resultados Treinamento - Config B:")
print("Min: ", min(Resultados_Treinamento_B))
print("Max: ", max(Resultados_Treinamento_B))
print("Media: ", mean(Resultados_Treinamento_B))
print("Std: ", std(Resultados_Treinamento_B))
plt.clf()
plt.boxplot(Resultados_Treinamento_B)
plt.title("Resultados Treinamento - Config B")
plt.savefig('Plots/Treinamento_B.png', format='png')
#plt.show()


archive.write("\nResultados Teste - Config A:")
archive.write("\nMin: " + str(min(Resultados_Teste_A)))
archive.write("\nMax: " + str(max(Resultados_Teste_A)))
archive.write("\nMedia: " + str(mean(Resultados_Teste_A)))
archive.write("\nStd: " + str(std(Resultados_Teste_A)))
print("Resultados Teste - Config A:")
print("Min: ", min(Resultados_Teste_A))
print("Max: ", max(Resultados_Teste_A))
print("Media: ", mean(Resultados_Teste_A))
print("Std: ", std(Resultados_Teste_A))
plt.clf()
plt.boxplot(Resultados_Teste_A)
plt.title("Resultados Teste - Config A")
plt.savefig('Plots/Teste_A.png', format='png')
#plt.show()

archive.write("\n\nResultados Teste - Config B:")
archive.write("\nMin: " + str(min(Resultados_Teste_B)))
archive.write("\nMax: " + str(max(Resultados_Teste_B)))
archive.write("\nMedia: " + str(mean(Resultados_Teste_B)))
archive.write("\nStd: " + str(std(Resultados_Teste_B)))
archive.write("\n----------------------")
print("Resultados Teste - Config B:")
print("Min: ", min(Resultados_Teste_B))
print("Max: ", max(Resultados_Teste_B))
print("Media: ", mean(Resultados_Teste_B))
print("Std: ", std(Resultados_Teste_B))
plt.clf()
plt.boxplot(Resultados_Teste_B)
plt.title("Resultados Teste - Config B")
plt.savefig('Plots/Teste_B.png', format='png')
#plt.show()

archive.write("\nMelhor resultado Teste - Config A:"+str(Melhor_Resultados_Teste_A))
archive.write("\nMelhor resultado Teste - Config B:"+str(Melhor_Resultados_Teste_B))
print("Melhor resultado Teste - Config A:", Melhor_Resultados_Teste_A)
print("Melhor resultado Teste - Config B:", Melhor_Resultados_Teste_B)

archive.close()

'''
print(Resultados_Treinamento_A)
print(Resultados_Treinamento_B)
print(Resultados_Teste_A)
print(Resultados_Teste_B)
'''
