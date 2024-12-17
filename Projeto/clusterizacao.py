import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
df_combined = pd.read_csv(r'C:\Users\ianli\OneDrive\Área de Trabalho\projeto 3\dataset-analysis\Data\winequality_combined.csv')
if 'Unnamed: 0' in df_combined.columns:
        df_combined= df_combined.drop(columns=['Unnamed: 0'])

# Exibir informações do dataframe
print(df_combined.head())
print(df_combined.info())
print(df_combined.columns)

# Pré-processamento: Remover a variável categórica 'type' ou codificar
df_combined = pd.get_dummies(df_combined, columns=['wine_type'], drop_first=True)

# Remover a variável alvo 'quality'
features = df_combined.drop(columns=['quality'])

# Escalando os dados
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Aplicação do K-Means com 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
df_combined['Cluster'] = kmeans.fit_predict(features_scaled)

# Visualização dos clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_combined['alcohol'], 
                y=df_combined['density'], 
                hue=df_combined['Cluster'], 
                palette="deep")
plt.title("Clusters Baseados em Álcool e Densidade")
plt.xlabel("Álcool")
plt.ylabel("Densidade")
plt.show()
