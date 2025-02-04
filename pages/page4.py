from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import seaborn as sns
from sklearn.tree import plot_tree  # Importação correta da função
import matplotlib.pyplot as plt
<<<<<<< Updated upstream
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Base de dados de vinhos
base_wine_red = pd.read_csv('dataset-analysis/Data/winequality-red.csv')
base_combined_wine= pd.read_csv('dataset-analysis/Data/winequality-red.csv')
base_wine_white = pd.read_csv('dataset-analysis/Data/df_white.csv')
=======
from sklearn.tree import plot_tree
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.cm as cm


# Base de dados de vinhos
base_wine_red = pd.read_csv('Data/winequality-red.csv')
base_combined_wine= pd.read_csv('Data/winequality-red.csv')
base_wine_white = pd.read_csv('Data/df_white.csv')
>>>>>>> Stashed changes


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
<<<<<<< Updated upstream
random_forest_wine_combined = RandomForestClassifier(n_estimators=100, random_state=0, criterion='entropy')
=======
random_forest_wine_combined = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=0, criterion='entropy')
>>>>>>> Stashed changes
random_forest_wine_white = RandomForestClassifier(n_estimators=100, random_state=0, criterion='entropy')


# Treinamento do modelo
random_forest_wine_red.fit(x_wine_red_treinamento, y_wine_red_treinamento)
random_forest_wine_combined.fit(x_wine_combined_treinamento, y_wine_combined_treinamento)
random_forest_wine_white.fit(x_wine_white_treinamento, y_wine_white_treinamento)


previsao_red= random_forest_wine_red.predict(x_wine_red_teste)
previsao_combined= random_forest_wine_combined.predict(x_wine_combined_teste)
previsao_white= random_forest_wine_white.predict(x_wine_white_teste)

print('Acurácia Vermelho:', accuracy_score(y_wine_red_teste, previsao_red))
print('Acurácia combinado:', accuracy_score(y_wine_combined_teste, previsao_combined))
print('Acurácia Branco:', accuracy_score(y_wine_white_teste, previsao_white))
#precisão de 67.29% na primeira tentativa do combined

<<<<<<< Updated upstream
plt.figure(figsize=(10, 6))
plot_tree(random_forest_wine_red.estimators_[0], filled=True, rounded=True, class_names=['3', '4', '5', '6', '7', '8'])
plt.show()
=======
#grafico da arvore de decisão
plt.figure(figsize=(20, 10))
plot_tree(random_forest_wine_combined.estimators_[0], 
        feature_names=['Acidez Fixa', 'Acidez Volátil', 'Ácido Cítrico', 'Açúcar', 'Clorides', 'Enxofre Livre', 'Enxofre Total', 'Densidade', 'pH', 'Sulfatos', 'Álcool'],
        filled=True, 
        fontsize=6, 
        rounded=True,
        class_names=[str(i) for i in range(3, 10)])



# Personalize o texto das caixas
for texto in plt.gca().texts:
    linhas = texto.get_text().split('\n')
    nova_linha = [linha for linha in linhas if not linha.startswith('value')]
    texto.set_text('\n'.join(nova_linha))



plt.title('Árvore de Decisão para o Modelo de Random Forest Combinado - Qualidade de Vinho', fontsize=18)
plt.tight_layout()  # Ajuste a layout para evitar que os rótulos sejam cortados
plt.savefig('arvore_combinada', dpi=300, bbox_inches='tight')  # Salva a imagem como 'arvore_decisao.png'
plt.show()
>>>>>>> Stashed changes
