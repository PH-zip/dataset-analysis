import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Título do aplicativo
st.title("Previsão da Qualidade do Vinho")

# Opção para escolher o modelo
modelo_escolhido = st.selectbox("Escolha o modelo para previsão:", ["Random Forest", "KNN", "XGBoost"])

# Formulário para entrada das variáveis
st.header("Insira os Valores das Variáveis")
fixed_acidity = st.number_input("Fixed Acidity", value=7.0)
volatile_acidity = st.number_input("Volatile Acidity", value=0.5)
citric_acid = st.number_input("Citric Acid", value=0.0)
residual_sugar = st.number_input("Residual Sugar", value=2.0)
chlorides = st.number_input("Chlorides", value=0.1)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=10.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=30.0)
density = st.number_input("Density", value=0.99)
pH = st.number_input("pH", value=3.0)
sulphates = st.number_input("Sulphates", value=0.5)
alcohol = st.number_input("Alcohol", value=10.0)

# Botão para fazer a previsão
if st.button("Prever Qualidade do Vinho"):
    # Criando um DataFrame com as entradas do usuário
    dados_usuario = pd.DataFrame({
        "fixed acidity": [fixed_acidity],
        "volatile acidity": [volatile_acidity],
        "citric acid": [citric_acid],
        "residual sugar": [residual_sugar],
        "chlorides": [chlorides],
        "free sulfur dioxide": [free_sulfur_dioxide],
        "total sulfur dioxide": [total_sulfur_dioxide],
        "density": [density],
        "pH": [pH],
        "sulphates": [sulphates],
        "alcohol": [alcohol]
    })

    try:
        # Carregando o scaler e o LabelEncoder
        scaler = joblib.load('scaler.pkl')
        label_quality = joblib.load('label_encoder.pkl')
        
        # Escalonando as entradas do usuário
        dados_usuario_scaled = scaler.transform(dados_usuario)
        
        # Escolhendo o modelo correspondente
        if modelo_escolhido == "Random Forest":
            modelo = joblib.load('modelo_RF.pkl')
        elif modelo_escolhido == "KNN":
            modelo = joblib.load('modelo_knn.pkl')
        elif modelo_escolhido == "XGBoost":
            modelo = joblib.load('xgboost_model.pkl')
        
        # Fazendo a previsão
        previsao = modelo.predict(dados_usuario_scaled)
        
        # Decodificando a previsão usando o LabelEncoder
        qualidade = label_quality.inverse_transform(previsao)
        
        # Exibindo o resultado
        st.success(f"A qualidade do vinho é: **{qualidade[0]}**")
    except FileNotFoundError:
        st.error("Arquivos do modelo não encontrados. Certifique-se de que os arquivos de modelo, scaler e label encoder estão no diretório correto.")
        
    #0=alto, 1=baixo, 2=médio
