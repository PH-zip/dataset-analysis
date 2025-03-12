import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Projeto.csv_to_parquet import conversor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib


def load_and_preprocess_data(df):
    df = df.drop(columns=['Unnamed: 0', 'tipo_vinho'], errors='ignore')
    
    # Tratando valores ausentes, se houver
    df.fillna(df.mean(), inplace=True)
    
    bins = [0, 5, 7, 10]
    labels = ['baixa', 'média', 'alta']
    df['categoria_qualidade'] = pd.cut(df['Qualidade'], bins=bins, labels=labels)
    
    # Convertendo as classes categóricas em numéricas
    le = LabelEncoder()
    df['categoria_qualidade'] = le.fit_transform(df['categoria_qualidade'])
    
    return df, le

def train_and_evaluate_model(df):
    X = df.drop(columns=['Qualidade', 'categoria_qualidade'])
    y = df['categoria_qualidade']
    
    # Divisão dos dados em conjunto de treino e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Ajuste de hiperparâmetros usando GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='f1_weighted')
    grid_search.fit(X_treino, y_treino)
    
    # Melhor modelo
    best_xgb = grid_search.best_estimator_
    
    # Previsões
    y_pred = best_xgb.predict(X_teste)
    
    # Acurácia
    accuracy = accuracy_score(y_teste, y_pred)
    
    joblib.dump(best_xgb, 'modelo_Xgboost.pkl')
    
    return accuracy, y_teste, y_pred

def main():
    # Configurações da aba
    st.set_page_config(
        page_title="Vinícola",  # Nome da aba
        page_icon=":wine_glass:",  # Emoji de vinho para a aba
    )

    # Caminho para o seu arquivo CSV e parquet
    red = conversor(R'Data/winequality-red.csv', R'Data/red.parquet')
    white = conversor(R'Data/df_white.csv', R'Data/white.parquet')

    # Ler arquivos parquet
    df_white = pd.read_parquet(white)
    df_red = pd.read_parquet(red)

    # Adicionar a coluna 'tipo_vinho'
    df_white['tipo_vinho'] = 'Branco'
    df_red['tipo_vinho'] = 'Tinto'

    # Unir os dois datasets
    combined_df = pd.concat([df_white, df_red])

    # Remover a coluna 'Unnamed: 0' caso exista
    if 'Unnamed: 0' in combined_df.columns:
        combined_df = combined_df.drop(columns=['Unnamed: 0'])

    # Renomeando as colunas para facilitar a análise
    novos_nomes = {
        "fixed acidity": "Acidez fixa",
        "volatile acidity": "Acidez volátil",
        "citric acid": "Ácido cítrico",
        "residual sugar": "Açúcar residual",
        "chlorides": "Cloretos",
        "free sulfur dioxide": "Dióxido de enxofre livre",
        "total sulfur dioxide": "Dióxido de enxofre total",
        "density": "Densidade",
        "pH": "pH",
        "sulphates": "Sulfatos",
        "alcohol": "Álcool",
        "quality": "Qualidade"
    }
    combined_df.rename(columns=novos_nomes, inplace=True)

            # Barra lateral com filtros
    st.sidebar.title("Aplique Filtros Aqui")
                # Slider para selecionar intervalo de pH
    # Slider para selecionar intervalo de pH
    ph_min, ph_max = st.sidebar.slider(
        "Selecione o intervalo de pH:",
        min_value=0.00,  # Valor mínimo permitido
        max_value=combined_df['pH'].max(),  # Valor máximo permitido
        value=(0.00, combined_df['pH'].max()),  # Intervalo inicial como tupla (min, max)
        step=0.1
    )

    # Slider para selecionar intervalo de teor alcoólico
    alcohol_min, alcohol_max = st.sidebar.slider(
        "Selecione o intervalo de teor alcoólico:",
        min_value=0.00,  # Valor mínimo permitido
        max_value=combined_df['Álcool'].max(),  # Valor máximo permitido
        value=(0.00, combined_df['Álcool'].max()),  # Intervalo inicial como tupla (min, max)
        step=0.1
    )

    # Slider para selecionar intervalo de teor de sulfatos
    sulfatos_min, sulfatos_max = st.sidebar.slider(
        "Selecione o intervalo de teor de sulfatos:",
        min_value=0.00,  # Valor mínimo permitido
        max_value=combined_df['Sulfatos'].max(),  # Valor máximo permitido
        value=(0.00, combined_df['Sulfatos'].max()),  # Intervalo inicial como tupla (min, max)
        step=0.1
    )

    # Slider para selecionar intervalo de teor de ácido cítrico
    acido_citrico_min, acido_citrico_max = st.sidebar.slider(
        "Selecione o intervalo de teor de ácido cítrico:",
        min_value=0.00,  # Valor mínimo permitido
        max_value=combined_df['Ácido cítrico'].max(),  # Valor máximo permitido
        value=(0.00, combined_df['Ácido cítrico'].max()),  # Intervalo inicial como tupla (min, max)
        step=0.01
    )

    # Slider para selecionar intervalo de teor de ácido clorídrico
    acido_cloridrico_min, acido_cloridrico_max = st.sidebar.slider(
        "Selecione o intervalo de teor de ácido clorídrico:",
        min_value=0.00,  # Valor mínimo permitido
        max_value=combined_df['Cloretos'].max(),  # Valor máximo permitido
        value=(0.00, combined_df['Cloretos'].max()),  # Intervalo inicial como tupla (min, max)
        step=0.01
    )

    # Slider para selecionar intervalo de teor de densidade
    densidade_min, densidade_max = st.sidebar.slider(
        "Selecione o intervalo de teor de densidade:",
        min_value=0.00,  # Valor mínimo permitido
        max_value=combined_df['Densidade'].max(),  # Valor máximo permitido
        value=(0.00, combined_df['Densidade'].max()),  # Intervalo inicial como tupla (min, max)
        step=0.001
    )



    # Slider para selecionar intervalo de teor de sulfatos totais
    sulfatos_totais_min, sulfatos_totais_max = st.sidebar.slider(
        "Selecione o intervalo de teor de sulfatos totais:",
        min_value=0.00,  # Valor mínimo permitido
        max_value=combined_df['Dióxido de enxofre total'].max(),  # Valor máximo permitido
        value=(0.00, combined_df['Dióxido de enxofre total'].max()),  # Intervalo inicial como tupla (min, max)
        step=0.01
    )


        # Checkbox para filtrar apenas vinhos tintos
    somente_vinhos_tintos = st.sidebar.checkbox("Apenas vinhos tintos")

        # Checkbox para filtrar apenas vinhos brancos
    somente_vinhos_brancos = st.sidebar.checkbox("Apenas vinhos brancos")

        # Aplicar filtros no DataFrame com base nos valores selecionados
    df_selecionado = combined_df[
        (combined_df['pH'] >= ph_min) & (combined_df['pH'] <= ph_max) & 
        (combined_df['Álcool'] >= alcohol_min) & (combined_df['Álcool'] <= alcohol_max) & 
        (combined_df['Sulfatos'] >= sulfatos_min) & (combined_df['Sulfatos'] <= sulfatos_max) & 
        (combined_df['Ácido cítrico'] >= acido_citrico_min) & (combined_df['Ácido cítrico'] <= acido_citrico_max) & 
        (combined_df['Cloretos'] >= acido_cloridrico_min) & (combined_df['Cloretos'] <= acido_cloridrico_max) & 
        (combined_df['Densidade'] >= densidade_min) & (combined_df['Densidade'] <= densidade_max) & 
        (combined_df['Dióxido de enxofre total'] >= sulfatos_totais_min) & (combined_df['Dióxido de enxofre total'] <= sulfatos_totais_max) 
        ]   

    # Filtrar por tipo de vinho se algum filtro foi ativado
    if somente_vinhos_tintos:
        df_selecionado = df_selecionado[df_selecionado['tipo_vinho'] == 'Tinto']
    elif somente_vinhos_brancos:
        df_selecionado = df_selecionado[df_selecionado['tipo_vinho'] == 'Branco']


    # Treinamento do modelo
    df_model = load_and_preprocess_data(df_selecionado)[0]
    accuracy, y_teste, y_pred = train_and_evaluate_model(df_model)
        
    st.header("Resultados do Modelo")
    st.markdown("---")

        # Matriz de confusão
    cm = confusion_matrix(y_teste, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['baixa', 'média', 'alta'], yticklabels=['baixa', 'média', 'alta'])
    plt.ylabel('Classe verdadeira')
    plt.xlabel('Classe prevista')
    plt.title('Matriz de Confusão')
    st.pyplot(plt)

    report = classification_report(y_teste, y_pred,  output_dict=True)
    df_report = pd.DataFrame(report).T
    st.write(df_report)

if __name__ == '__main__':
    main()
