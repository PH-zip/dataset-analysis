import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np

# Configuração inicial do Streamlit
st.set_page_config(page_title="Análise de Cluster de Vinhos")
st.title("Análise de Cluster de Vinhos 🍷")

st.markdown("---")

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
n_clusters = st.sidebar.slider("Número de Clusters", 3, 4, 5,)

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
st.pyplot(fig)  

st.markdown("---")

# --- Método do Cotovelo ---
st.subheader("Método do Cotovelo para Determinar o Melhor Número de Clusters")

sse = []
k_range = range(1, 11)  # Testa valores de k de 1 a 10
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    sse.append(kmeans.inertia_)  # inertia_ retorna o SSE

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_range, sse, marker='o')
ax.set_xlabel('Número de Clusters (k)')
ax.set_ylabel('Soma dos Erros Quadráticos (SSE)')
ax.set_title('Método do Cotovelo')
st.pyplot(fig)

#coeficiente de silhueta
silhouette = silhouette_score(features_scaled, df_combined['Cluster'])

# Calcular os valores de silhueta para cada ponto
sample_silhouette_values = silhouette_samples(features_scaled, df_combined['Cluster'])

st.markdown("---")

# --- Gráfico de Silhueta ---
st.subheader("Análise de Silhueta")

fig, ax = plt.subplots(figsize=(10, 6))

y_lower = 10
for i in range(n_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[df_combined['Cluster'] == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = plt.cm.nipy_spectral(float(i) / n_clusters)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10
ax.set_title("Gráfico de Silhueta")
ax.set_xlabel("Coeficiente de Silhueta")
ax.set_ylabel("Cluster")
ax.axvline(x=silhouette, color="red", linestyle="--")

ax.set_yticks([])
ax.set_xticks(np.arange(-0.1, 1.1, 0.2))

st.pyplot(fig)

# --- Gráfico Estruturado com Grade (Conforme Imagem) ---
st.subheader("Visualização dos Clusters com Grade Estruturada")

fig, ax = plt.subplots(figsize=(10, 6))

# Definir grades conforme a imagem
x_ticks = [0, 2, 4, 6, 8, 10, 12]       # Feature 1st (eixo X)
y_ticks = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8]  # Feature 2nd (eixo Y)

# Plot dos clusters com grade
sns.scatterplot(
    x=df_combined[x_axis],
    y=df_combined[y_axis],
    hue=df_combined['Cluster'],
    palette="viridis",
    ax=ax,
    s=50,
    edgecolor='w'
)

# Configurações da grade
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.grid(True, linestyle='--', alpha=0.6, which='both')
ax.set_xlim(-1, 13)   # Limites do eixo X
ax.set_ylim(-11, 9)   # Limites do eixo Y

# Rótulos e título
ax.set_title("Distribuição dos Clusters nos Espaços das Features", pad=15)
ax.set_xlabel("Feature 1 (Espaço Estruturado)", fontsize=10)
ax.set_ylabel("Feature 2 (Espaço Estruturado)", fontsize=10)

# Legenda
ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

st.pyplot(fig)


st.markdown("---")

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

st.markdown("---")

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


