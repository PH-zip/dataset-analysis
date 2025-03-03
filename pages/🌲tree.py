from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import seaborn as sns
from sklearn.tree import plot_tree  # Importação correta da função
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st


# Base de dados de vinhos
base_wine_red = pd.read_csv('Data/winequality-red.csv')
base_combined_wine= pd.read_csv('Data/winequality-red.csv')
base_wine_white = pd.read_csv('Data/df_white.csv')


# Selecione todas as linhas e colunas, exceto a última coluna (que é a qualidade)
x_wine_red = base_wine_red.iloc[:, :-1]
x_wine_combined = base_combined_wine.iloc[:, :-1]
x_wine_white = base_wine_white.iloc[:, :-1]

# Selecione todas as linhas e apenas a última coluna (que é a qualidade)

y_wine_red = base_wine_red.iloc[:, -1]
y_wine_combined = base_combined_wine.iloc[:, -1]
y_wine_white = base_wine_white.iloc[:, -1]



# Normalização dos dados

scaler_wine = StandardScaler()

x_wine_red = scaler_wine.fit_transform(x_wine_red)
x_wine_combined = scaler_wine.fit_transform(x_wine_combined)
x_wine_white = scaler_wine.fit_transform(x_wine_white)


# Dividindo a base de dados em treinamento e teste

x_wine_red_treinamento, x_wine_red_teste, y_wine_red_treinamento, y_wine_red_teste =train_test_split(x_wine_red, y_wine_red, test_size=0.3, random_state=0)
x_wine_combined_treinamento, x_wine_combined_teste, y_wine_combined_treinamento, y_wine_combined_teste =train_test_split(x_wine_combined, y_wine_combined, test_size=0.3, random_state=0)
x_wine_white_treinamento, x_wine_white_teste, y_wine_white_treinamento, y_wine_white_teste =train_test_split(x_wine_white, y_wine_white, test_size=0.3, random_state=0)


print('Base de treinamento:', x_wine_red_treinamento.shape)
print('Base de teste:', x_wine_red_teste.shape)
print('Base de treinamento:', x_wine_combined_treinamento.shape) 


# Salvando os dados em um arquivo

with open('wine_red.pkl', mode='wb') as f:
    pickle.dump([x_wine_red_treinamento, y_wine_red_treinamento, x_wine_red_teste, y_wine_red_teste], f)

with open('wine_combined.pkl', mode='wb') as f:
    pickle.dump([x_wine_combined_treinamento, y_wine_combined_treinamento, x_wine_combined_teste, y_wine_combined_teste], f)

with open('wine_white.pkl', mode='wb') as f:
    pickle.dump([x_wine_white_treinamento, y_wine_white_treinamento, x_wine_white_teste, y_wine_white_teste], f)    



# Criação do modelo

random_forest_wine_red = RandomForestClassifier(n_estimators=100, random_state=0, criterion='entropy')
random_forest_wine_combined = RandomForestClassifier(n_estimators=100, random_state=0, criterion='entropy')
random_forest_wine_white = RandomForestClassifier(n_estimators=100, random_state=0, criterion='entropy')


# Treinamento do modelo
random_forest_wine_red.fit(x_wine_red_treinamento, y_wine_red_treinamento)
random_forest_wine_combined.fit(x_wine_combined_treinamento, y_wine_combined_treinamento)
random_forest_wine_white.fit(x_wine_white_treinamento, y_wine_white_treinamento)


previsao_red= random_forest_wine_red.predict(x_wine_red_teste)
previsao_combined= random_forest_wine_combined.predict(x_wine_combined_teste)
previsao_white= random_forest_wine_white.predict(x_wine_white_teste)

acuracia_red =  accuracy_score(y_wine_red_teste, previsao_red)
acuracia_combined = accuracy_score(y_wine_combined_teste, previsao_combined)
acuracia_white = accuracy_score(y_wine_white_teste, previsao_white)
#precisão de 67.29% na primeira tentativa do combined

st.subheader("Resultados da Classificação")

st.write(f"**Acurácia Vinho Tinto:** {acuracia_red:.2%}")
st.write(f"**Acurácia Vinho Combinado:** {acuracia_combined:.2%}")
st.write(f"**Acurácia Vinho Branco:** {acuracia_white:.2%}")