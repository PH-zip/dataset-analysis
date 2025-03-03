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

        # Título e introdução
        st.title("Relacoes")

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
        st.markdown("---")

        # Criar uma tabela de contingência para ambos os tipos de vinho
        heatmap_data = combined_df.pivot_table(values='Acidez volátil', 
                                                index='Qualidade', 
                                                columns='tipo_vinho', 
                                                aggfunc='mean')

        # Criar o heatmap para ambos os tipos de vinho
        plt.figure(figsize=(10, 6), facecolor="lightgray")
        sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
        plt.title('Qualidade X Acidez Volátil', fontsize=18, color='black')
        plt.xlabel('Acidez Volátil', fontsize=14)
        plt.ylabel('Qualidade', fontsize=14)
        st.pyplot(plt)  # Use st.pyplot para exibir o gráfico no Streamlit
        st.caption("Vinhos de qualidade inferior geralmente apresentam níveis mais elevados de acidez volátil.")
        
        
        st.markdown("---")
        # Criar o gráfico de linha
        
        plt.figure(figsize=(10, 6), facecolor='lightgray')  # Cor de fundo da figura
        ax = sns.lineplot(data=combined_df, x='Qualidade', y='Álcool', hue='tipo_vinho', marker='o', palette={'Branco': 'darkgray', 'Tinto': 'darkred'})

        # Adicionar título e rótulos
        plt.title('Relação entre Álcool e Qualidade dos Vinhos', fontsize=16)
        plt.xlabel('Qualidade', fontsize=14)
        plt.ylabel('Teor Alcoólico', fontsize=14)
        plt.xticks(range(3, 9))  # Ajustar os ticks do eixo x para as qualidades esperadas
        plt.xlim(3, 8)  # Limitar o eixo x para que vá de 3 a 8
        plt.grid(True)
        st.pyplot(plt)
        st.caption("Vinhos com maior teor alcoólico geralmente são associados a sabores mais intensos e equilibrados.")
          # Exibir o gráfico no Streamlit
        st.markdown("---")

        # Lmplot para Vinhos Brancos
        # Combinar os DataFrames de vinhos brancos e tintos
        combined_df = pd.concat([df_white, df_red])

        # Lmplot combinado
        plt.figure(figsize=(10, 6), facecolor='lightgray')
        lmplot = sns.lmplot(data=combined_df, x='Qualidade', y='Densidade', 
                        hue='tipo_vinho', palette={'Branco': 'lightgray', 'Tinto': 'darkred'}, 
                        height=6, aspect=1.5)

        # Título e rótulos
        lmplot.figure.suptitle('Qualidade X Densidade - Vinhos Brancos e Tintos', fontsize=18)
        lmplot.set_axis_labels('Qualidade', 'Densidade')
        st.pyplot(lmplot.figure)
        st.caption( st.caption("Vinhos de alta qualidade geralmente apresentam uma densidade mais baixa, devido ao teor alcoólico mais elevado e à menor quantidade de açúcar residual"))

if __name__ == '__main__':
        main() 
