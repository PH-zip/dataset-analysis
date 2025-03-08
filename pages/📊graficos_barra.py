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
    
    st.sidebar.image("logo_vinho.jpg",  use_container_width=True )

    
    st.title("Graficos de barras")
    
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


    # Gráfico de barras agrupado: Ácido cítrico x Qualidade
    st.markdown("---")
    st.subheader("Comparação do Ácido cítrico por Qualidade entre Vinhos Tintos e Brancos")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
    data=df_selecionado, 
    x='Qualidade', 
    y='Ácido cítrico', 
    hue='tipo_vinho', 
    palette={"Branco": "lightgray", "Tinto": "darkred"}
)
    ax.set_xlabel('Qualidade')
    ax.set_ylabel('Ácido cítrico')
    ax.legend(title='Tipo de Vinho', loc='upper right')
    st.pyplot(fig)
    st.caption("Vinhos Tintos: a acidez tende a aumentar confrome a qualidade percebida")
    st.caption("Vinhos Brancos: a acidez tende a se manter estavel")
    

    # Gráfico de barras - Cloretos x Qualidade
    st.markdown("---")
    st.subheader("Cloretos x Qualidade")

    # Agrupar e calcular a média dos cloretos
    cloretos_df = df_selecionado.groupby(['Qualidade', 'tipo_vinho'])['Cloretos'].mean().unstack()

    # Criar o gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 6))
    cloretos_df.plot(kind='bar', color={'Branco': 'lightgray', 'Tinto': 'darkred'}, ax=ax)
    ax.set_xlabel('Qualidade')
    ax.set_ylabel('Cloretos')
    ax.set_title('Média dos Cloretos por Qualidade e Tipo de Vinho')
    ax.legend(title='Tipo de Vinho', labels=['Branco', 'Tinto'])
    st.pyplot(fig)
    st.caption("Vinhos Tintos: a variacao do nivel de cloretos nao tende a afetar muito a qualidade final do vinho, porem, uma concentracao muito alta(0,12) eh percebida como uma qualidade inferior")
    st.caption("Vinhos Brancos: a variacao dos cloretos tende a ser estavel, porem diminuindo um pouco conforme a qualidade aumenta")
if __name__ == '__main__':
    main() 