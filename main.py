import streamlit as st
import pandas as pd

def main():
    
    #Isso é só para a aba o navegador ficar com nome e icone personalizados
    st.set_page_config(page_title="Vinicola", page_icon=":wine_glass:", layout="wide", initial_sidebar_state="collapsed")

    
    #Aqui é o titulo do nosso dataset e alguns tipos de textos para usar no dataset
    st.title("Análise de qualidade de vinhos")
    st.subheader("lorum ipsum, apenas testando os paragrafos usando streamlit")
    st.markdown("**tenhamos preferencia para usar o markdown**, ele permite formatar o *texto*")

    st.markdown("---") #divisoria
    
    # Caminho para o seu arquivo CSV
    dataset_caminho = 'D:\PH\GitHub\dataset-analysis\winequality-red.csv'  # Substitua pelo caminho do seu arquivo CSV


    # Ler o arquivo CSV usando pandas
    try:
        df = pd.read_csv(dataset_caminho)

        # Exibir o DataFrame no Streamlit
        st.write("Dados do Dataset:")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Ocorreu um erro ao ler o arquivo: {e}")
        
        
    #Barra lateral com os filtros
    
    st.markdown("---") #divisoria
    
    st.sidebar.title("Local para Aplicar Filtros")
    
    ph = st.sidebar.slider("Seleciona o PH minimo:", min_value=0.0, max_value=14.0, value=3.0, step=0.1)
    
    df.selecionado= df.query("pH >= @ph")
    
    st.dataframe(df.selecionado)
    
if __name__ == "__main__":
    main()