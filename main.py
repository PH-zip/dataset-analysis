import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from csv_to_parquet import conversor

def main():
    # Configurações da aba
    st.set_page_config(
        page_title="Vinícola",  # Nome da aba
        page_icon=":wine_glass:",  # Emoji de vinho para a aba
    )

    # Título e introdução
    st.title("Análise de Qualidade de Vinhos")
    st.markdown("---")  # Divisória

    # Caminho para o seu arquivo CSV e parquet
    dataset_caminho = conversor(R'C:\Users\ianli\OneDrive\Área de Trabalho\projeto 3\dataset-analysis\Data\winequality-red.csv' , R'C:\Users\ianli\OneDrive\Área de Trabalho\projeto 3\dataset-analysis\Data\dataset.parquet')  # Substitua pelo caminho correto

    # Ler o arquivo parquet usando pandas
    try:
        df = pd.read_parquet(dataset_caminho)
        st.write(f"### Dados do Dataset ({len(df)} linhas)")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Ocorreu um erro ao tentar ler o arquivo: {e}")
        return  # Sai da função se houver erro no carregamento

    st.markdown("---")

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
    df_selecionado = df[
    (df['pH'] >= ph_min) & (df['pH'] <= ph_max) &
    (df['alcohol'] >= alcohol_min) & (df['alcohol'] <= alcohol_max)
]
    #botao de apenas vinhos secos
    somente_vinhos_secos = st.sidebar.checkbox("Apenas vinhos secos")
    if somente_vinhos_secos:
     df_selecionado = df_selecionado[df_selecionado['residual sugar'] <= 4]

# Exibir os dados filtrados
    st.write(f"### Dados Filtrados ({len(df_selecionado)} linhas)")
    st.dataframe(df_selecionado)

    # Gráfico de barras - Distribuição da qualidade
    st.markdown("---")
    st.subheader("Distribuição com base na Qualidade dos Vinhos")
    qualidade_counts = df_selecionado['quality'].value_counts().sort_index()
    fig, ax = plt.subplots()
    qualidade_counts.plot(kind='bar', color='darkred', ax=ax)
    ax.set_xlabel('Qualidade')
    ax.set_ylabel('Quantidade')
    st.pyplot(fig)

    # Gráfico de dispersão - pH vs álcool
    st.markdown("---")
    st.subheader("Gráfico de Dispersão: pH vs Teor Alcoólico")
    fig, ax = plt.subplots()
    ax.set_facecolor('lightgray')  # Fundo do gráfico
    sns.scatterplot(data=df_selecionado, x='pH', y='alcohol', ax=ax, color='darkred')
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
    coluna = st.selectbox("Selecione a coluna para análise", options=df.columns)
    contagem = df_selecionado[coluna].value_counts().sort_index()
    st.bar_chart(contagem)

if __name__ == '__main__':
    main() 