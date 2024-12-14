import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from csv_to_parquet import conversor
from leitor_df import leitor
def main():
    # Configurações da aba
    st.set_page_config(
        page_title="Vinícola",  # Nome da aba
        page_icon=":wine_glass:",  # Emoji de vinho para a aba
    )

    # Título e introdução
    st.title("Análise da Qualidade de Vinhos: Explorando correlacao entre variaveis")

    # Caminho para o seu arquivo CSV e parquet
    red_path = conversor(R'C:\Users\ianli\OneDrive\Área de Trabalho\projeto 3\dataset-analysis\Data\winequality-red.csv' , R'C:\Users\ianli\OneDrive\Área de Trabalho\projeto 3\dataset-analysis\Data\red.parquet')  # Substitua pelo caminho do seu pc
    white_path = conversor(R'C:\Users\ianli\OneDrive\Área de Trabalho\projeto 3\dataset-analysis\Data\winequality-white.csv', R'C:\Users\ianli\OneDrive\Área de Trabalho\projeto 3\dataset-analysis\Data\white.parquet')# Substitua pelo caminho do seu pc
    
    # Ler o arquivo parquet
    df_red = leitor(red_path)
    df_white = leitor(white_path)

    #Renomeando as colunas para facilitar a análise
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
        "quality": "Qualidade"}
    df_red.rename(columns=novos_nomes, inplace=True)
    df_white.rename(columns=novos_nomes,inplace=True)
    st.write(f"### Dados do Dataset ({len(df_red)} linhas)")
    st.dataframe(df_red)

    # Barra lateral com filtros
    st.sidebar.title("Local para Aplicar Filtros")

# Slider para selecionar intervalo de pH
    ph_min, ph_max = st.sidebar.slider(
    "Selecione o intervalo de pH:",
    min_value=2.74,  # Valor mínimo permitido
    max_value=4.01,  # Valor máximo permitido
    value=(2.74, 4.01),  # Intervalo inicial como tupla (min, max)
    step=0.1
)

# Slider para selecionar intervalo de teor alcoólico
    alcohol_min, alcohol_max = st.sidebar.slider(
    "Selecione o intervalo de teor alcoólico:",
    min_value=8.4,  # Valor mínimo permitido
    max_value=15.0,  # Valor máximo permitido
    value=(8.4, 15.0),  # Intervalo inicial como tupla (min, max)
    step=0.1
)

# Aplicar o filtro no DataFrame usando os intervalos
    df_selecionado = df_red[
    (df_red['pH'] >= ph_min) & (df_red['pH'] <= ph_max) &
    (df_red['alcohol'] >= alcohol_min) & (df_red['alcohol'] <= alcohol_max)
]
    #botao de apenas vinhos tintos
    somente_vinhos_tintos = st.sidebar.checkbox("Apenas vinhos tintos")
    if somente_vinhos_tintos:
     df_selecionado = df_red

    #botao vinhos brancos
    somente_vinhos_brancos = st.sidebar.checkbox("Apenas vinhos brancos")
    if somente_vinhos_brancos:
       df_selecionado = df_white
        


    # Gráfico de barras - Distribuição da qualidade
    st.markdown("---")
    st.subheader("Distribuição com base na Qualidade dos Vinhos")
    qualidade_counts = df_selecionado['quality'].value_counts().sort_index()
    fig, ax = plt.subplots()
    qualidade_counts.plot(kind='bar', color='darkred', ax=ax)
    ax.set_xlabel('Qualidade')
    ax.set_ylabel('Quantidade')
    st.pyplot(fig)

    # Gráfico de dispersão - pH vs qualidade
    st.markdown("---")
    st.subheader("Gráfico de Dispersão: pH vs Qualidade")
    fig, ax = plt.subplots()
    ax.set_facecolor('lightgray')  # Fundo do gráfico
    sns.scatterplot(data=df_selecionado, x='pH', y='quality', ax=ax, color='darkred')
    st.pyplot(fig)

    # Boxplot - Teor alcoólico por qualidade
    st.markdown("---")
    st.subheader("Boxplot do Teor Alcoólico por Qualidade")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df_selecionado, x='quality', y='alcohol', palette='Set2', ax=ax)
    ax.set_xlabel('Qualidade')
    ax.set_ylabel('Teor Alcoólico')
    st.pyplot(fig)

    # Estatísticas descritivas
    st.markdown("---")
    st.subheader("Estatísticas Descritivas")
    st.write(df_selecionado.describe())

    # Distribuição de variáveis
    st.markdown("---")
    st.subheader("Distribuição de Variáveis")
    coluna = st.selectbox("Selecione a coluna para análise", options=df_red.columns)
    contagem = df_selecionado[coluna].value_counts().sort_index()
    st.bar_chart(contagem)

    

if __name__ == '__main__':
    main() 