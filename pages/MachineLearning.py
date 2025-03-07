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

def main():
    # Configurações da aba
    st.set_page_config(
        page_title="Vinícola",  # Nome da aba
        page_icon=":wine_glass:",  # Emoji de vinho para a aba
    )
    st.title("Machine Learning")
    st.header("Rain Forest")

    # Lendo o arquivo csv
    df = pd.read_csv('Data\winequality_combined.csv')

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

    joblib.dump(RF, 'random_forest_model.pkl')  # Salva o modelo treinado
    joblib.dump(scaler, 'scaler.pkl')  # Salva o scaler
    joblib.dump(label_quality, 'label_encoder.pkl') 


    # Escolhendo a árvore
    estimator = RF.estimators_[0]

    # Plotar a árvore de decisão
    fig, ax = plt.subplots(figsize=(50, 30), dpi=100)
    plot_tree(estimator,
                feature_names=X.columns,  # Nomes das features
                class_names=labels,  # Nomes das classes
                filled=True,
                rounded=True,
                proportion=True,
                max_depth=2)  # Limitando a profundidade
    st.pyplot(fig)
    

    
    # Exibindo a matriz de confusão como tabela
    report = classification_report(y_teste, X_previsao_RF,  output_dict=True)
    df_report = pd.DataFrame(report).T
    st.write(df_report)



    #----------------------------------------------------------------------------------------
    st.markdown("---")

    st.subheader("KNN")

        
    X_treino_knn = scaler.fit_transform(X_treino)
    X_teste_knn = scaler.transform(X_teste)

    taxa_erro = []

    for i in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(X_treino_knn, y_treino_knn)
        pred_i = knn.predict(X_teste_knn)
        taxa_erro.append(np.mean(pred_i != y_teste))

    plt.figure(figsize =(20, 12))
    plt.plot(range(1, 21), taxa_erro,linestyle ='dashed', marker ='o',	markerfacecolor ='red', markersize = 10)

    plt.title('Taxa_erro por Geração')
    plt.xlabel('Geração')
    plt.ylabel('Taxa_erro')
    st.pyplot(plt)

    knn_final = KNeighborsClassifier(n_neighbors = 18)

    knn_final.fit(X_treino_knn, y_treino_knn)
    pred_final = knn_final.predict(X_teste_knn)

    print(accuracy_score(y_teste, pred_final))

    print(classification_report(y_teste, pred_final))
    
    with open('modelo_KNN.pkl', 'wb') as arquivo:
        joblib.dump(knn_final, arquivo)
    
if __name__ == '__main__':
    main()
