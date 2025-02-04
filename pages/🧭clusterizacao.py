import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração inicial do Streamlit
st.set_page_config(page_title="Análise de Cluster de Vinhos", layout="wide")
st.title("Análise de Cluster de Vinhos 🍷")

# Carregar dados
@st.cache_data
def load_data():
    df = pd.read_csv(r'Data\winequality_combined.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df

df_combined = load_data()

# Pré-processamento
@st.cache_data
def preprocess_data(df):
    df_processed = pd.get_dummies(df, columns=['wine_type'], drop_first=True)
    features = df_processed.drop(columns=['quality'])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled, features

df_combined['wine_type'] = df_combined['wine_type'].replace({
    'red': 0,
    'white': 1
})

features_scaled, features = preprocess_data(df_combined)

# Sidebar para controles interativos
st.sidebar.header("Configurações dos Clusters")
n_clusters = st.sidebar.slider("Número de Clusters", 2, 5, 3)

# Seleção de variáveis para visualização
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox("Eixo X", features.columns, index=features.columns.get_loc('alcohol'))
with col2:
    y_axis = st.selectbox("Eixo Y", features.columns, index=features.columns.get_loc('density'))

# Aplicar K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_combined['Cluster'] = kmeans.fit_predict(features_scaled)

# Criar visualização
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    x=df_combined[x_axis],
    y=df_combined[y_axis],
    hue=df_combined['Cluster'],
    palette="deep",
    ax=ax
)
ax.set_title(f"Clusters Baseados em {x_axis} e {y_axis}")
ax.set_xlabel(x_axis)
ax.set_ylabel(y_axis)

# Exibir gráfico no Streamlit
st.pyplot(fig)

# Mostrar dados estatísticos
st.subheader("Estatísticas por Cluster")

# Selecionar colunas numéricas (incluindo 'Cluster' se for numérica)
numeric_cols = df_combined.select_dtypes(include='number').columns.tolist()

# Garantir que 'Cluster' está nas colunas numéricas
if 'Cluster' not in numeric_cols:
    numeric_cols.append('Cluster')

# Calcular estatísticas
cluster_stats = df_combined[numeric_cols].groupby('Cluster').mean()

# Exibir com formatação
st.dataframe(
    cluster_stats.style.format("{:.2f}")
    .background_gradient(cmap='Blues', axis=0)
    .set_properties(**{'text-align': 'center'})
)
