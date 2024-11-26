import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    
    #Configurações da aba
    st.set_page_config(
        page_title="Vinicola",  # Nome da aba
        page_icon=":wine_glass:",  # Emoji de vinho para a aba
        )
    
    # Aqui é o título do nosso dataset e alguns tipos de textos para usar no dataset
    st.title("Análise de qualidade de vinhos")
    st.markdown("---")  # divisoria
    
    # Caminho para o seu arquivo CSV
    dataset_caminho = 'D:\PH\GitHub\dataset-analysis\Data\winequality-red.csv'  # Substitua pelo caminho do seu arquivo CSV

    # Ler o arquivo CSV usando pandas
    try:
        df = pd.read_csv(dataset_caminho)



        # Isso exibe o DataFrame no Streamlit, e caso tenha algum erro ele vai mostrar a linha
        
        # Barra lateral com os filtros
        st.sidebar.title("Local para Aplicar Filtros")
        
        # filtro de arrastar que o usuário usa pra selecionar o valor mínimo de pH
        ph = st.sidebar.slider("Seleciona o PH minimo:", min_value=0.0, max_value=14.0, value=3.0, step=0.1)
        
        alcool = st.sidebar.slider("Seleciona o teor alcoolico maximo:", min_value=0.0, max_value=15.0, value=9.0, step=0.1)
        
        acidez = st.sidebar.slider("Seleciona o teor de acidez maximo:", min_value=0.0, max_value=20.0, value=19.0, step=0.1)
        # Filtrar dados do DataFrame com base no valor de pH selecionado
        
        df.selecionado = df.query("pH >= @ph and alcohol <= @alcool") 
        
        st.subheader("Dados Filtrados")
        st.dataframe(df.selecionado)
        
        
        
        
        # Criar um gráfico de colunas da quantidade de vinhos por qualidade
        st.markdown("---")  # divisória
        st.subheader("Distribuiçãocom base na Qualidade dos Vinhos")

        # Contar a quantidade de vinhos por qualidade
        qualidade_counts = df['quality'].value_counts()

            # Criar um gráfico de barras com Matplotlib
        fig, ax = plt.subplots()
        
        qualidade_counts.plot(kind='bar', color='darkred', ax=ax)  # Plota o gráfico de barras com cores personalizadas

        ax.set_xlabel('Qualidade') #rotulo do eixo x
        ax.set_ylabel('Quantidade') #rotulo do eixo y

        # Exibir o gráfico no Streamlit
        st.pyplot(fig)
                
        
        
        st.markdown("---")  
        st.subheader("Início da análise")
        
        # Criar um gráfico de dispersão e mostra a figura de comparação
        st.subheader("Gráfico de Dispersão: pH vs Alcool")
        fig, ax = plt.subplots()
        
        ax.set_facecolor('lightgray') #testando colorir o fundo do gráfico para ficar mais visivel
        sns.scatterplot(data=df, x='pH', y='alcohol', ax=ax, color='darkred')
        st.pyplot(fig)
        
        
        
        st.subheader("Boxplot do Teor Alcoólico por Qualidade")
        
        plt.figure(figsize=(10, 6)) # Aqui é o tamanho do grafico em polegadas
        sns.boxplot(x='quality', y='alcohol', data=df, palette='Set2')
        plt.xlabel('Qualidade')
        plt.ylabel('Teor Alcoólico')

        # Exibir o boxplot no Streamlit
        st.pyplot(plt)
        
        
        st.write(df.describe())

                
        
        st.markdown('---')

    except Exception as e:
        st.error(f"Ocorreu um erro ao tentar ler o arquivo: {e}")

if __name__ == '__main__': 
    main()