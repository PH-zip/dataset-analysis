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

    plt.figure(figsize=(10, 6), facecolor='lightgray')  # Cor de fundo da figura
    ax = sns.lineplot(data=combined_df, x='Qualidade', y='pH', hue='tipo_vinho', marker='o', palette={'Branco': 'darkgray', 'Tinto': 'darkred'})

        # Adicionar título e rótulos
    plt.title('Relação entre Ph e Qualidade dos Vinhos', fontsize=16)
    plt.xlabel('Qualidade', fontsize=14)
    plt.ylabel('Ph', fontsize=14)
    plt.xticks(range(3, 9))  # Ajustar os ticks do eixo x para as qualidades esperadas
    plt.xlim(3, 8)  # Limitar o eixo x para que vá de 3 a 8
    plt.grid(True)
    st.pyplot(plt)
    st.caption("Vinhos Tintos: um pH mais baixo está associado a uma qualidade percebida como mais alta")
    st.caption("Vinhos Brancos: um pH mais alto pode estar relacionado a uma qualidade percebida como mais alta")

    # Gráfico de linha - Açúcar Residual vs Qualidade
    st.markdown("---")
    st.subheader("Gráfico de Linha: Açúcar Residual vs Qualidade por Tipo de Vinho")

    # Criar um DataFrame para o gráfico de linha
    plt.figure(figsize=(10, 6), facecolor='lightgray')  # Cor de fundo da figura
    ax = sns.lineplot(data=combined_df, x='Qualidade', y='Açúcar residual', hue='tipo_vinho', marker='o', palette={'Branco': 'darkgray', 'Tinto': 'darkred'})

        # Adicionar título e rótulos
    plt.title('Relação entre Açúcar Residual e Qualidade dos Vinhos', fontsize=16)
    plt.xlabel('Qualidade', fontsize=14)
    plt.ylabel('Açúcar Residual', fontsize=14)
    plt.xticks(range(3, 9))  # Ajustar os ticks do eixo x para as qualidades esperadas
    plt.xlim(3, 8)  # Limitar o eixo x para que vá de 3 a 8
    plt.grid(True)
    st.pyplot(plt)
    st.caption("Vinhos Tintos: o acucar residual tende a se manter estavel nao afetando muito na percepção de qualidade")
    st.caption("Vinhos Brancos: a percepção de qualidade pode estar relacionada a niveis moderados de acucar , porem tentendo a diminuir conforme a qualidade aumenta ")
    

    # Gráfico de linhas - Dióxido de Enxofre Livre x Qualidade
    st.markdown("---")
    st.subheader("Dióxido de Enxofre Livre x Qualidade")

    # Agrupar e calcular a média
    plt.figure(figsize=(10, 6), facecolor='lightgray')  # Cor de fundo da figura
    ax = sns.lineplot(data=combined_df, x='Qualidade', y='Dióxido de enxofre livre', hue='tipo_vinho', marker='o', palette={'Branco': 'darkgray', 'Tinto': 'darkred'})

        # Adicionar título e rótulos
    plt.title('Relação entre Dióxido de enxofre livre e Qualidade dos Vinhos', fontsize=16)
    plt.xlabel('Qualidade', fontsize=14)
    plt.ylabel('Dióxido de enxofre livre', fontsize=14)
    plt.xticks(range(3, 9))  # Ajustar os ticks do eixo x para as qualidades esperadas
    plt.xlim(3, 8)  # Limitar o eixo x para que vá de 3 a 8
    plt.grid(True)
    st.pyplot(plt)
    st.caption("Vinhos Tintos: um menor nível de dióxido de enxofre livre também pode estar associado a uma qualidade percebida como mais alta")
    st.caption("Vinhos Brancos: o nivel de dioxido de enxofre tende a ser estavel, porem niveis muito altos so mostram associados a uma qualidade percebida menor")

    # Gráfico de linhas -  Acidez fixa x Qualidade
    st.markdown("---")
    st.subheader("Acidez fixa x Qualidade")

    plt.figure(figsize=(10, 6), facecolor='lightgray')  # Cor de fundo da figura
    ax = sns.lineplot(data=combined_df, x='Qualidade', y='Acidez fixa', hue='tipo_vinho', marker='o', palette={'Branco': 'darkgray', 'Tinto': 'darkred'})

        # Adicionar título e rótulos
    plt.title('Relação entre Acidez fixa e Qualidade dos Vinhos', fontsize=16)
    plt.xlabel('Qualidade', fontsize=14)
    plt.ylabel('Acidez fixa', fontsize=14)
    plt.xticks(range(3, 9))  # Ajustar os ticks do eixo x para as qualidades esperadas
    plt.xlim(3, 8)  # Limitar o eixo x para que vá de 3 a 8
    plt.grid(True)
    st.pyplot(plt)
    st.caption("Vinhos Tintos: O ponto ideal de acidez pode variar mas tende a quanto mais alto melhor a qualidade percebida")
    st.caption("Vinhos Brancos: um menor nível de acidez fixa está associado a uma qualidade percebida como mais alta")

    
if __name__ == '__main__':
     main() 