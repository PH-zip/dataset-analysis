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

    # Exibindo a distribuição das classes antes do SMOTE
    st.subheader("Distribuição das Classes")
    st.write("`__________________`**0 = Alta, 1 = Baixa, 2 = Média**`__________________`")

    # Criando colunas para exibir as tabelas lado a lado
    col1, col2 = st.columns(2)

    # Tabela antes do SMOTE
    with col1:
        st.markdown("**Antes do SMOTE**")
        contagem_antes = pd.Series(y_treino).value_counts().sort_index()
        st.write(contagem_antes)

    # Aplicando o SMOTE para balancear a classe alta, que possui menos exemplares
    smote = SMOTE(random_state=0)
    X_treino_balanceado, y_treino_balanceado = smote.fit_resample(X_treino, y_treino)

    # Tabela pra comparação
    with col2:
        st.markdown("**Após o SMOTE**")
        contagem_depois = pd.Series(y_treino_balanceado).value_counts().sort_index()
        st.write(contagem_depois)

    # Normalizando os dados
    scaler = StandardScaler()
    X_treino_knn = scaler.fit_transform(X_treino_balanceado)
    X_teste_knn = scaler.transform(X_teste)

    # Treinando o modelo KNN
    knn_final = KNeighborsClassifier(n_neighbors=1)
    knn_final.fit(X_treino_knn, y_treino_balanceado)

    # Fazendo previsões
    pred_final = knn_final.predict(X_teste_knn)

    # Avaliando o modelo
    st.subheader("Visualização:")

    # Exibindo a acurácia
    acuracia = accuracy_score(y_teste, pred_final)
    st.write(f"Acurácia: {acuracia:.2f}")

    
    # Binarizar as classes para multiclasse
    y_teste_bin = label_binarize(y_teste, classes=[0, 1, 2])
    pred_proba = knn_final.predict_proba(X_teste_knn)

    # Calcular a curva ROC para cada classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):  # 3 classes: baixa, média, alta
        fpr[i], tpr[i], _ = roc_curve(y_teste_bin[:, i], pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plotar a curva ROC
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, label=f'Classe {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC-KNN')
    plt.legend(loc="lower right")
    st.pyplot(plt)
    
    
    
    
    
    #joblib.dump(knn_final, 'Modelo_KNN')   

if __name__ == '__main__':
    main()
