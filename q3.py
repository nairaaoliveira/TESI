import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from model.sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB

#TREINANDO ALGORITMOS APLICAVEIS (MultinomialNB, AdaBoost, OneVsRest, OneVsOne, OutputCode)
dt = pd.read_csv('forestfires.csv')	
meses = dt[['X', 'Y','month','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']]
dias = dt['day']

conv_meses = pd.get_dummies(meses).astype(int)

M = conv_meses.values
D = dias.values

porcentagem_treino = 0.8
porcentagem_teste = 0.2

tamanho_de_treino = int(porcentagem_treino * len(D))
tamanho_de_teste = int(porcentagem_teste * len(D))

treino_dados = M[:tamanho_de_treino]
treino_marcacoes = D[:tamanho_de_treino]

fim_de_treino = tamanho_de_treino + tamanho_de_teste

teste_dados = M[tamanho_de_teste:fim_de_treino]
teste_marcacoes = D[tamanho_de_teste:fim_de_treino]

validacao_dados = M[fim_de_treino:]
validacao_marcacoes = D[fim_de_treino:]

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    modelo.fit(treino_dados, treino_marcacoes)
    resultado = modelo.predict(teste_dados)
    acertos = 0
    tamanho = len(teste_marcacoes)

    for i in range(tamanho):
        if teste_marcacoes[i] == resultado[i]:
            acertos = acertos + 1

    print('Acerto %s: %.2f%%' % (nome, (acertos* 100/ tamanho)))
    print("Total de acertos: ", acertos)
    

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomialNB = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state = 0))
resultadoOneVsRest = fit_and_predict("OneVsRest", modeloOneVsRest, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

from sklearn.multiclass import OneVsOneClassifier
modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state = 0))
resultadoOneVsOne = fit_and_predict("OneVsOne", modeloOneVsOne, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

from sklearn.multiclass import OutputCodeClassifier
modeloOutputCode = OutputCodeClassifier(LinearSVC(random_state = 0))
resultadoOutputCode = fit_and_predict("OutputCode", modeloOutputCode, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)


#K-FOLDING
print("Com o K-Folding:\n")
dt = pd.read_csv('forestfires.csv')	
#meses = dt[['X', 'Y','month']]
meses = dt[['X', 'Y','month','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']]
dias = dt['day']

conv_meses = pd.get_dummies(meses).astype(int)
M = conv_meses.values
D = dias.values

porcentagem_treino = 0.9
tamanho_treino = int(porcentagem_treino * len(D))
tamanho_validacao = len(D) - tamanho_treino

dados_treino = M[:tamanho_treino]
marcacoes_treino = D[:tamanho_treino]

dados_teste = M[-tamanho_validacao:]
marcacoes_teste = D[-tamanho_validacao:]

from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()
k = 6
scores = cross_val_score(modelo,dados_treino,marcacoes_treino,cv = k)
print(scores)
taxa_de_acerto = np.mean(scores)
print("Taxa de acerto (MultinomialNB): ", round((taxa_de_acerto *100),2), "%")

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
k = 6
scores = cross_val_score(modeloAdaBoost,dados_treino,marcacoes_treino,cv = k)
print(scores)
taxa_de_acerto = np.mean(scores)
print("Taxa de acerto (AdaBoost): ", round((taxa_de_acerto *100),2), "%")

from sklearn.multiclass import OneVsRestClassifier
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state = 0))
k = 6
scores = cross_val_score(modeloOneVsRest,dados_treino,marcacoes_treino,cv = k)
print(scores)
taxa_de_acerto = np.mean(scores)
print("Taxa de acerto (OneVsRest): ", round((taxa_de_acerto *100),2), "%")

from sklearn.multiclass import OneVsOneClassifier
modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state = 0))
k = 6
scores = cross_val_score(modeloOneVsOne,dados_treino,marcacoes_treino,cv = k)
print(scores)
taxa_de_acerto = np.mean(scores)
print("Taxa de acerto (OneVsOne): ", round((taxa_de_acerto *100),2), "%")

from sklearn.multiclass import OutputCodeClassifier
modeloOutputCode = OutputCodeClassifier(LinearSVC(random_state = 0))
k = 6
scores = cross_val_score(modeloOutputCode,dados_treino,marcacoes_treino,cv = k)
print(scores)
taxa_de_acerto = np.mean(scores)
print("Taxa de acerto (OutputCode): ", round((taxa_de_acerto *100),2), "%")
