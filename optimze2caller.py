import pandas as pd

df = pd.read_csv('input_tobb_haztartas.csv', sep=';', index_col=0, parse_dates=True)
df_filtered = df.resample("1h").sum()
na_values = df_filtered.values

p_consumed=df_filtered[['consumer1','consumer2']].values
p_pv=df_filtered[['pv1','pv2']].values
p_ut=df_filtered[['thermal_user1','thermal_user2']].values

print(p_consumed)
print(p_pv)
print(p_ut)
#print(na_values)
#print(df.resample("1h").mean().values)