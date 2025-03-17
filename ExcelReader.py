import pandas as pd
import os

# TODO:
# input.csv-ból kiszedni a negyedórás adatokat:
# PV termelését
# Háztartás fogyasztását
# Kiolvasni, dataframe-ben eltárolod
#Commit1

df = pd.read_csv('input.csv')
df_filtered = df[~df.iloc[:, 0].astype(str).str.contains(":15|:30|:45", regex=True)]
print(df_filtered)
