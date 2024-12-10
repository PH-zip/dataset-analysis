import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Configurações da aba
    st.set_page_config(
        page_title="Vinícola",  # Nome da aba
        page_icon=":wine_glass:",  # Emoji de vinho para a aba
    )

    # Título e introdução
    st.title("Análise de Qualidade de Vinhos")
    st.markdown("---")  # Divisória

    # Caminho para o seu arquivo CSV
    dataset_caminho = 'C:/Users/ianli/OneDrive/Área de Trabalho/projeto 3/dataset-analysis/Data/winequality-red.csv'  # Substitua pelo caminho correto

    # Ler o arquivo CSV usando pandas
    try:
        df = pd.read_csv(dataset_caminho)
        st.write("### Dados do Dataset")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Ocorreu um erro ao tentar ler o arquivo: {e}")
        return  # Sai da função se houver erro no carregamento

    st.markdown("---")

    # Barra lateral com filtros
    st.sidebar.title("Local para Aplicar Filtros")
    ph = st.sidebar.slider("Seleciona o pH mínimo:", min_value=0.0, max_value=14.0, value=3.0, step=0.1)
    alcool = st.sidebar.slider("Seleciona o teor alcoólico máximo:", min_value=0.0, max_value=15.0, value=9.0, step=0.1)

    # Filtrar os dados
    df_selecionado = df.query("pH >= @ph and alcohol <= @alcool")
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