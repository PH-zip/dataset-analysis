import pandas as pd
#Renomeando as colunas para facilitar a análise
white = pd.read_csv(R"C:\Users\jefee\OneDrive\Área de Trabalho\FACULDADE UFRPE\PROJETO 3\dataset-analysis-main\INFORMAÇÕES DE DADOS DOS VINHOS", sep= ";")
white.to_csv('df_white.csv', sep=',')
