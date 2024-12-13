import pandas as pd
import streamlit as st
def leitor(path): 
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        st.error(f"Ocorreu um erro ao tentar ler o arquivo: {e}")
        return  # Sai da função se houver erro no carregamento
