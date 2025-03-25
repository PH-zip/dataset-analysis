import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import joblib
from imblearn.combine import SMOTEENN  # Substituindo SMOTE por SMOTEENN
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from PIL import Image
from sklearn.metrics import precision_recall_curve, average_precision_score


def main():
    # Configurações da aba
    st.set_page_config(
        page_title="Vinícola",
        page_icon=":wine_glass:"
    )

    st.sidebar.image("logo_vinho.jpg", use_container_width=True)
    st.title("Machine Learning")
    st.header("Rain Forest")

    # Lendo o dataset
    df = pd.read_csv('Data/winequality_combined.csv')
    df = df.drop(columns=['Unnamed: 0', 'wine_type'])

    # Criando a nova coluna categórica
    bins = [0, 5, 7, 10]
    labels = ['baixa', 'média', 'alta']
    df['categoria_qualidade'] = pd.cut(df['quality'], bins=bins, labels=labels)

    # Separando variáveis independentes e target
    X = df.drop(columns=['quality', 'categoria_qualidade'])
    y = df['categoria_qualidade']

    # Codificando o target
    label_quality = LabelEncoder()
    y = label_quality.fit_transform(y)

    # Dividindo o dataset
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalonamento
    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino)
    X_teste = scaler.transform(X_teste)
    X_treino_RF = scaler.fit_transform(X_treino)
    X_teste_RF = scaler.transform(X_teste)
    
    # Aplicando SMOTEENN
    st.subheader("Balanceamento de Classes com SMOTEENN")
    smoteenn = SMOTEENN(random_state=42)
    X_treino_balanceado, y_treino_balanceado = smoteenn.fit_resample(X_treino, y_treino)

    st.write("Distribuição das classes após o SMOTEENN:")
    st.write(pd.Series(y_treino_balanceado).value_counts())

    # Usando hiperparâmetros fixos no modelo
    parametros_melhores = {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 5, 'n_estimators': 10}

    melhor_RF = RandomForestClassifier(**parametros_melhores, random_state=0)
    melhor_RF.fit(X_treino_balanceado, y_treino_balanceado)

    # Previsões
    X_previsao_RF = melhor_RF.predict(X_teste)
    X_previsao_treino_RF = melhor_RF.predict(X_treino_balanceado)

    # Relatórios
    st.subheader("Relatório de Teste - Random Forest (Melhor Modelo)")
    report_teste_RF = classification_report(y_teste, X_previsao_RF, output_dict=True, target_names=labels)
    df_report_teste_RF = pd.DataFrame(report_teste_RF).T
    st.write(df_report_teste_RF)

    st.subheader("Relatório de Treino - Random Forest (Melhor Modelo)")
    report_treino_RF = classification_report(y_treino_balanceado, X_previsao_treino_RF, output_dict=True, target_names=labels)
    df_report_treino_RF = pd.DataFrame(report_treino_RF).T
    st.write(df_report_treino_RF)

    # Matriz de confusão
    st.subheader("Matriz de Confusão - Random Forest (Melhor Modelo)")
    matriz_confusao_RF = confusion_matrix(y_treino_balanceado, X_previsao_treino_RF)
    df_matriz_confusao_RF = pd.DataFrame(matriz_confusao_RF)
    st.write(df_matriz_confusao_RF)

    # Exibindo imagem da árvore
    st.header("Árvore de Decisão - Random Forest🌳 (Melhor Modelo)")
    try:
        arvore_image = Image.open('arvore_decisao_random_forest.png')
        st.image(arvore_image, caption='Árvore de Decisão - Random Forest', use_container_width=True)
    except FileNotFoundError:
        st.error("Imagem da árvore de decisão não encontrada. Gere a imagem primeiro.")

    #----------------------------------------------------------------------------------------
    from imblearn.over_sampling import SMOTE

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




    # Curva ROC (imagem)
    st.header("Curva ROC - KNN")
    try:
        # Exibindo a acurácia
        acuracia = accuracy_score(y_teste, pred_final)
        st.write(f"Acurácia: {acuracia:.2f}")
        roc_image = Image.open('curva_roc_knn.png')
        st.image(roc_image, use_container_width=True)
    except FileNotFoundError:
        st.error("Imagem da curva ROC não encontrada. Gere a imagem primeiro.")

if __name__ == '__main__':
    main()