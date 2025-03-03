import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import shap

# Configuração para exibir gráficos no notebook
%matplotlib inline

# Função principal
def main():
    # Carregar dados
    df = pd.read_csv('Data/winequality_combined.csv')
    df = df.drop(columns=['Unnamed: 0', 'wine_type'])

    # Processamento de dados
    bins = [0, 5, 7, 10]
    labels = ['baixa', 'média', 'alta']
    df['categoria_qualidade'] = pd.cut(df['quality'], bins=bins, labels=labels)

    X = df.drop(columns=['quality', 'categoria_qualidade'])
    y = df['categoria_qualidade']

    # Codificação e divisão dos dados
    label_quality = LabelEncoder()
    y = label_quality.fit_transform(y)
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pré-processamento
    scaler = StandardScaler()
    X_treino_RF = scaler.fit_transform(X_treino)
    X_teste_RF = scaler.transform(X_teste)

    # Modelo Random Forest
    RF = RandomForestClassifier(n_estimators=250, random_state=42)
    RF.fit(X_treino_RF, y_treino)
    X_previsao_RF = RF.predict(X_teste_RF)

    # Salvar modelos
    joblib.dump(RF, 'random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_quality, 'label_encoder.pkl')

    # Visualização da árvore
    print("\nVisualização da Árvore de Decisão:")
    estimator = RF.estimators_[0]
    plt.figure(figsize=(50, 30), dpi=100)
    plot_tree(estimator,
              feature_names=X.columns,
              class_names=labels,
              filled=True,
              rounded=True,
              proportion=True,
              max_depth=2)
    plt.show()

    # Métricas de avaliação
    print("\nRelatório de Classificação:")
    print(classification_report(y_teste, X_previsao_RF))

    # Modelo KNN
    print("\nTreinando Modelo KNN:")
    X_treino_knn = scaler.fit_transform(X_treino)
    X_teste_knn = scaler.transform(X_teste)

    taxa_erro = []
    for i in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_treino_knn, y_treino)
        pred_i = knn.predict(X_teste_knn)
        taxa_erro.append(np.mean(pred_i != y_teste))

    plt.figure(figsize=(20, 12))
    plt.plot(range(1, 21), taxa_erro, linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
    plt.title('Taxa de Erro por Geração')
    plt.xlabel('Geração')
    plt.ylabel('Taxa de Erro')
    plt.show()

    knn_final = KNeighborsClassifier(n_neighbors=18)
    knn_final.fit(X_treino_knn, y_treino)
    joblib.dump(knn_final, 'modelo_KNN.pkl')

    # Análise SHAP
    print("\nAnalisando SHAP Values:")
    try:
        explainer = shap.TreeExplainer(RF)
        X_teste_df = pd.DataFrame(X_teste_RF, columns=X.columns)
        shap_values = explainer.shap_values(X_teste_df)

        # Summary plot
        print("\nSHAP Summary Plot:")
        shap.summary_plot(shap_values, X_teste_df)
        plt.show()

        # Dependence plots por classe
        print("\nSHAP Dependence Plots:")
        for i, classe in enumerate(labels):
            print(f"\nClasse: {classe}")
            if 'alcohol' in X.columns:
                shap.dependence_plot(
                    "alcohol",
                    shap_values[i],
                    X_teste_df,
                    show=False
                )
                plt.tight_layout()
                plt.show()
                
    except Exception as e:
        print(f"Erro na geração dos gráficos SHAP: {str(e)}")

    # Importância das features
    print("\nImportância das Variáveis:")
    importances = RF.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Importância das Variáveis")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
