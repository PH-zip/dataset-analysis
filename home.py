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

    # Título e introdução
    st.title("Análise da Qualidade de Vinhos: Explorando correlação entre variáveis")

    # Caminho para o seu arquivo CSV e parquet
    red = conversor(R'Data/winequality-red.csv', R'Data/red.parquet')
    white = conversor(R'Data/df_white.csv', R'Data/white.parquet')

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
    # Exibir dados do dataset
    st.write(f"### Dados do Dataset ({len(df_selecionado)} linhas)")
    st.dataframe(df_selecionado)

    # Gráfico de barras - Distribuição da qualidade
    st.markdown("---")
    st.subheader("Distribuição com base na Qualidade dos Vinhos")
    qualidade_counts = df_selecionado['Qualidade'].value_counts().sort_index()
    fig, ax = plt.subplots()
    qualidade_counts.plot(kind='bar', color='darkred', ax=ax)
    ax.set_xlabel('Qualidade')
    ax.set_ylabel('Quantidade') 
    # Rotacionar os ticks no eixo X (usando ax.tick_params)
    ax.tick_params(axis='x', rotation=0)
    st.pyplot(fig)

    # Estatísticas descritivas
    st.markdown("---")
    st.subheader("Estatísticas Descritivas")
    st.write(df_selecionado.describe())

    # Distribuição de variáveis
    st.markdown("---")
    st.subheader("Distribuição de Variáveis")
    coluna = st.selectbox("Selecione a coluna para análise", options=combined_df.columns)

    # Calcular a contagem
    contagem = df_selecionado[coluna].value_counts().reset_index()
    contagem.columns = [coluna, 'count']  # Renomeia as colunas

    # Exibir o gráfico
    st.bar_chart(contagem.set_index(coluna)['count'])

if __name__ == '__main__':
    main() 
