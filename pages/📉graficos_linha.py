import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Projeto.csv_to_parquet import conversor

def main():
    # Configurações da aba
    st.set_page_config(
        page_title="Vinícola",  # Nome da aba
        page_icon=":wine_glass:",  # Emoji de vinho para a aba
    )
    st.title("Graficos de linhas")

    # Caminho para o seu arquivo CSV e parquet
    red = conversor(R'Data\winequality-red.csv', R'Data\red.parquet')
    white = conversor(R'Data\df_white.csv', R'Data\white.parquet')


    # Ler arquivos parquet
    df_white = pd.read_parquet(white)
    df_red = pd.read_parquet(red)

    # Adicionar a coluna 'wine_type'
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
    df_white.rename(columns=novos_nomes, inplace=True)
    df_red.rename(columns=novos_nomes, inplace=True)

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

    # Checkbox para filtrar apenas vinhos tintos
    somente_vinhos_tintos = st.sidebar.checkbox("Apenas vinhos tintos")

    # Checkbox para filtrar apenas vinhos brancos
    somente_vinhos_brancos = st.sidebar.checkbox("Apenas vinhos brancos")

    # Aplicar filtros no DataFrame com base nos valores selecionados
    df_selecionado = combined_df[
        (combined_df['pH'] >= ph_min) & (combined_df['pH'] <= ph_max) &
        (combined_df['Álcool'] >= alcohol_min) & (combined_df['Álcool'] <= alcohol_max)
    ]

    # Filtrar por tipo de vinho se algum filtro foi ativado
    if somente_vinhos_tintos:
        df_selecionado = df_selecionado[df_selecionado['tipo_vinho'] == 'Tinto']
    elif somente_vinhos_brancos:
        df_selecionado = df_selecionado[df_selecionado['tipo_vinho'] == 'Branco']

    # Gráfico de linha - pH vs Qualidade por Tipo de Vinho
    st.markdown("---")
    st.subheader("Gráfico de Linha: pH vs Qualidade por Tipo de Vinho")

    # Criar um DataFrame para o gráfico de linha
    line_chart_data = df_selecionado.groupby(['Qualidade', 'tipo_vinho'])['pH'].mean().unstack().reset_index()

    fig, ax = plt.subplots()
    line_chart_data.set_index('Qualidade').plot(ax=ax, color={'Branco': 'lightgray', 'Tinto': 'darkred'})
    ax.set_title("Média do pH por Qualidade dos Vinhos")
    ax.set_xlabel("Qualidade")
    ax.set_ylabel("pH")
    st.pyplot(fig)
    st.caption("Vinhos Tintos: um pH mais baixo está associado a uma qualidade percebida como mais alta")
    st.caption("Vinhos Brancos: um pH mais alto pode estar relacionado a uma qualidade percebida como mais alta")

    # Gráfico de linha - Açúcar Residual vs Qualidade
    st.markdown("---")
    st.subheader("Gráfico de Linha: Açúcar Residual vs Qualidade por Tipo de Vinho")

    # Criar um DataFrame para o gráfico de linha
    line_chart_data = df_selecionado.groupby(['Qualidade', 'tipo_vinho'])['Açúcar residual'].mean().unstack().reset_index()

    # Exibir o gráfico de linha com cores específicas
    fig, ax = plt.subplots()
    line_chart_data.set_index('Qualidade').plot(ax=ax, color={'Branco': 'lightgray', 'Tinto': 'darkred'})
    ax.set_title("Média do Açúcar Residual por Qualidade dos Vinhos")
    ax.set_xlabel("Qualidade")
    ax.set_ylabel("Açúcar Residual")
    st.pyplot(fig)  
    st.caption("Vinhos Tintos: o acucar residual tende a se manter estavel nao afetando muito na percepção de qualidade")
    st.caption("Vinhos Brancos: a percepção de qualidade pode estar relacionada a niveis moderados de acucar , porem tentendo a diminuir conforme a qualidade aumenta ")
    

    # Gráfico de linhas - Dióxido de Enxofre Livre x Qualidade
    st.markdown("---")
    st.subheader("Dióxido de Enxofre Livre x Qualidade")

    # Agrupar e calcular a média
    so2_df = df_selecionado.groupby(['Qualidade', 'tipo_vinho'])['Dióxido de enxofre livre'].mean().unstack()

    # Criar o gráfico de linhas
    fig, ax = plt.subplots(figsize=(10, 6))
    so2_df.plot(kind='line', marker='o', color={'Branco': 'lightgray', 'Tinto': 'darkred'}, ax=ax)
    ax.set_xlabel('Qualidade')
    ax.set_ylabel('Dióxido de Enxofre Livre')
    ax.set_title('Média do Dióxido de Enxofre Livre por Qualidade e Tipo de Vinho')
    ax.legend(title='Tipo de Vinho', labels=['Branco', 'Tinto'])
    st.pyplot(fig)
    st.caption("Vinhos Tintos: um menor nível de dióxido de enxofre livre também pode estar associado a uma qualidade percebida como mais alta")
    st.caption("Vinhos Brancos: o nivel de dioxido de enxofre tende a ser estavel, porem niveis muito altos so mostram associados a uma qualidade percebida menor")

    # Gráfico de linhas -  Acidez fixa x Qualidade
    st.markdown("---")
    st.subheader("Acidez fixa x Qualidade")

    # Agrupar e calcular a média
    so2_df = df_selecionado.groupby(['Qualidade', 'tipo_vinho'])['Acidez fixa'].mean().unstack()

    # Criar o gráfico de linhas
    fig, ax = plt.subplots(figsize=(10, 6))
    so2_df.plot(kind='line', marker='o', color={'Branco': 'lightgray', 'Tinto': 'darkred'}, ax=ax)
    ax.set_xlabel('Qualidade')
    ax.set_ylabel('Acidez fixa')
    ax.set_title(' Media da Acidez fixa por Qualidade e Tipo de Vinho')
    ax.legend(title='Tipo de Vinho', labels=['Branco', 'Tinto'])
    st.pyplot(fig)
    st.caption("Vinhos Tintos: O ponto ideal de acidez pode variar mas tende a quanto mais alto melhor a qualidade percebida")
    st.caption("Vinhos Brancos: um menor nível de acidez fixa está associado a uma qualidade percebida como mais alta")

    
if __name__ == '__main__':
    main() 