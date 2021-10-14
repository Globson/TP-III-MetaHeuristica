import pandas as pd

def le_arq():
    tabela = pd.read_csv("../Data/telecom_users.csv")
    print(tabela.info())
    tabela.drop(columns='Unnamed: 0', inplace=True, axis=1)
    tabela["TotalGasto"] = pd.to_numeric(tabela["TotalGasto"], errors="coerce")
    tabela.dropna(how="all", axis=1,inplace=True)
    tabela.dropna(how="any", axis=0,inplace=True)
    print(tabela.info())
    return tabela
