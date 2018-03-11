import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

dt = pd.read_csv('forestfires.csv')	
dt.head()

#TIPOS DAS COLUNAS
print("\nVARIAVEL  -  TIPO\n", dt.dtypes)

#CAMPOS NULOS
print("\nNulos\n ", dt.isnull().sum())
#print("\nNulos\n ", dt.isnull().any())

#NOVA COLUNA chuva_area
dt['chuva_area'] = (dt['rain'] * dt['area'])
print("\n\n", dt.drop(columns=['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']))

#CONVERSAO DAS VARIAVEIS CATEGORICAS PARA INTEIRO
from collections import Counter
meses = dt[['X', 'Y','month']]
conv_meses = pd.get_dummies(meses).astype(int)

print("\nConversao de dados:\n", conv_meses)

#NOVA ANALISE DE DADOS
import nltk

dt = pd.read_csv('forestfires.csv')

entidades = nltk.pos_tag(dt['month'])
print(entidades)
