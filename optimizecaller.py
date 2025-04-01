import pandas as pd
from matplotlib import pyplot as plt

from optimize import optimize

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
na_values = df_filtered.values  # numpy array
# Optimize meghívása ezekkel az értékekkel
# Valamilyen megjelenítése az erdményeknek
#print(df_filtered)

results, status, objective, num_vars, num_constraints = optimize(na_values[:, 0],na_values[:, 1],na_values[:, 2],size_elh=4, bess_size=20)
print(results)
print(status)
print(objective)


time_index = df.resample("1h").sum().index

p_pv = na_values[:,0]


plt.figure(figsize=(12, 6))
plt.plot(time_index.hour, p_pv[0:24], label="PV Production", linestyle='-', marker='o')
plt.plot(time_index.hour, results["p_ue"].iloc[0:24], label="Electricity Used", linestyle='--', marker='x')
plt.plot(time_index.hour, results["p_with"], label="Withdrawn Power", linestyle='-.', marker='s')
plt.plot(time_index.hour, results["p_inj"], label="Injected Power", linestyle=':', marker='d')
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.title("Optimization Results Over Time")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

