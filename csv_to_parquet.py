import pandas as pd
import streamlit as st


def conversor(csv_path,parquet_path):
    try:
        # Ler o arquivo CSV
        df = pd.read_csv(csv_path)
        # Salvar como Parquet
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as e:
        st.error(f"Erro ao converter arquivo: {e}")
        return None