import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração inicial do Streamlit
st.set_page_config(page_title="Análise de Cluster de Vinhos")
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

# --- Scatter Plot ---
fig, ax = plt.subplots(figsize=(10, 6))  # Reduzindo ainda mais o tamanho

sns.scatterplot(
    x=df_combined[x_axis],
    y=df_combined[y_axis],
    hue=df_combined['Cluster'],
    palette="deep",
    ax=ax
)

ax.set_title(f"Clusters Baseados em {x_axis} e {y_axis}", fontsize=10)
ax.set_xlabel(x_axis, fontsize=12)
ax.set_ylabel(y_axis, fontsize=12)

fig.tight_layout()  # Ajusta os espaçamentos
st.pyplot(fig)  # Exibe no Streamlit sem esticar

# --- Heatmap ---
st.subheader("Qualidade Média por Cluster")

fig, ax = plt.subplots(figsize=(10, 6))  # Bem menor

heatmap_data = df_combined.groupby('Cluster')['quality'].mean().to_frame()

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=.5,
    ax=ax
)

ax.set_title('Média de Qualidade por Cluster', fontsize=10, pad=5)
ax.set_xlabel('Qualidade Média', fontsize=12)
ax.set_ylabel('Cluster', fontsize=12)

fig.tight_layout()  # Ajusta espaçamentos para evitar excesso de espaço
st.pyplot(fig)  # Sem "use_container_width"


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


