import pandas as pd
#Renomeando as colunas para facilitar a análise
white = pd.read_csv(R'C:\Users\ianli\OneDrive\Área de Trabalho\projeto 3\dataset-analysis\Data\winequality-white.csv', sep= ';')
white.to_csv('df_white.csv', sep=',')

