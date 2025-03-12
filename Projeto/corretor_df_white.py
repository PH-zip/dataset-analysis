import pandas as pd
#Renomeando as colunas para facilitar a análise
white = pd.read_csv(R'Data/winequality-white.csv', sep= ';')
white.to_csv('df_white.csv', sep=',')

