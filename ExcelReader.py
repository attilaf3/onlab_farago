import pandas as pd

# Ctrl+Shift+Alt+L: code cleanup: hosszú sorok tördelése+importok optimalizálása
# TODO:
# input.csv-ból kiszedni a negyedórás adatokat:
# PV termelését
# Háztartás fogyasztását
# Kiolvasni, dataframe-ben eltárolod
# Commit1
# %%
df = pd.read_csv('input.csv', sep=';', index_col=0, parse_dates=True)
# df_filtered = df[~df.index.astype(str).str.contains(":15|:30|:45", regex=True)]  # df.resample("1h").first()
df_filtered = df.resample("1h").sum()
df_filtered.values  # numpy array
# Optimize meghívása ezekkel az értékekkel
# Valamilyen megjelenítése az erdményeknek
print(df_filtered)
