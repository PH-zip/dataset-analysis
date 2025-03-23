import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from PIL import Image 
from sklearn.metrics import precision_recall_curve, average_precision_score


def main():
    # Configurações da aba
    st.set_page_config(
        page_title="Vinícola",  # Nome da aba
        page_icon=":wine_glass:",  # Emoji de vinho para a aba
    )
    st.sidebar.image("logo_vinho.jpg",  use_container_width=True )
    
    
    st.title("Machine Learning")
    st.header("Random Forest🌳")

    # Lendo o arquivo csv
    df = pd.read_csv('Data/winequality_combined.csv')

    # Tirando as colunas de texto que bugam o dataset
    df = df.drop(columns=['Unnamed: 0', 'wine_type'])

    bins = [0, 5, 7, 10]  # Definindo os intervalos
    labels = ['baixa', 'média', 'alta']  # Divisão das categorias
    df['categoria_qualidade'] = pd.cut(df['quality'], bins=bins, labels=labels)

    X = df.drop(columns=['quality', 'categoria_qualidade'])
    y = df['categoria_qualidade']

    print(df['categoria_qualidade'].value_counts())

    label_quality = LabelEncoder()

    # Codificando a variável target
    label_quality = LabelEncoder()
    y = label_quality.fit_transform(y)

    # Dividindo o dataset em treino e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

        # Escalonando as features
    scaler = StandardScaler()
    X_treino_RF = scaler.fit_transform(X_treino)
    X_teste_RF = scaler.transform(X_teste)

    y_treino_RF=y_treino
    y_treino_knn=y_treino

    # Criando o modelo de classificação
    RF = RandomForestClassifier(n_estimators=250, random_state=42)
    RF.fit(X_treino_RF, y_treino_RF)

    X_previsao_RF = RF.predict(X_teste_RF)

    print(classification_report(y_teste, X_previsao_RF))  # 84% de precisão

    #Necessario somente uma vez para criar o modelo, deixado como comentario para evitar lentidão
    #joblib.dump(RF, 'random_forest_model.pkl') 
    #joblib.dump(scaler, 'scaler.pkl')  
    #joblib.dump(label_quality, 'label_encoder.pkl') 


    # Escolhendo a árvore
    estimator = RF.estimators_[0]

    # Plotar a árvore de decisão
    st.header("Árvore de Decisão - Random Forest🌳")
    try:
        arvore_image = Image.open('arvore_decisao_random_forest.png')
        st.image(arvore_image, caption='Árvore de Decisão - Random Forest', use_container_width=True)
    except FileNotFoundError:
        st.error("Imagem da árvore de decisão não encontrada. Gere a imagem primeiro.")

    # Exibir a curva ROC do KNN
    st.header("Curva ROC - KNN")
    try:
        roc_image = Image.open('curva_roc_knn.png')
        st.image(roc_image, caption='Curva ROC - KNN', use_container_width=True)
    except FileNotFoundError:
        st.error("Imagem da curva ROC não encontrada. Gere a imagem primeiro.")

    
    
    
    
    #joblib.dump(knn_final, 'Modelo_KNN')   

if __name__ == '__main__':
    main()
