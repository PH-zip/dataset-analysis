import streamlit as st
import pandas as pd

def main():
    
    
    #Aqui é o titulo do nosso dataset e alguns tipos de textos para usar no dataset
    st.title("Análise de qualidade de vinhos")
    st.subheader("lorum ipsum, apenas testando os paragrafos usando streamlit")
    st.markdown("**tenhamos preferencia para usar o markdown**, ele permite formatar o *texto*")

    st.markdown("---") #divisoria
    
    # Caminho para o seu arquivo CSV
    dataset_caminho = 'D:\PH\GitHub\dataset-analysis\Data\winequality-red.csv'  # Substitua pelo caminho do seu arquivo CSV


    # Ler o arquivo CSV usando pandas
    try:
        df = pd.read_csv(dataset_caminho)

        # Isso exibe o DataFrame no Streamlit, e caso tenha algum erro ele vai mostrar a linha
        st.write("Dados do Dataset:")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Ocorreu um erro ao ler o arquivo: {e}")
        


    st.markdown("---") #divisoria
    
    
    
    #Barra lateral com os filtros
    st.sidebar.title("Local para Aplicar Filtros")
    
    # filtro de arrastar que o usuário usa pra selecionar o valor mínimo de pH
    ph = st.sidebar.slider("Seleciona o PH minimo:", min_value=0.0, max_value=14.0, value=3.0, step=0.1)
    
    alcool=st.sidebar.slider("Seleciona o teor alcoolico maximo:", min_value=0.0, max_value=15.0, value=9.0, step=0.1)
    
    
    # Filtrar dados do DataFrame com base no valor de pH selecionado
    df.selecionado= df.query("pH >= @ph and alcohol <= @alcool") 
    
    
    st.dataframe(df.selecionado)
    
    
    # Criar um gráfico de colunas da quantidade de vinhos por qualidade
    st.markdown("---")  # divisória
    st.subheader("Distribuição da Qualidade dos Vinhos")

    # Contar a quantidade de vinhos por qualidade
    qualidade_counts = df['quality'].value_counts()

    # Exibir o gráfico de barras usando Streamlit
    st.bar_chart(qualidade_counts)
    
    st.write(df.describe())
    
if __name__ == "__main__":
    main()