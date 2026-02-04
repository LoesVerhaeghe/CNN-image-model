import pandas as pd
import matplotlib.pyplot as plt
import os

df=pd.read_csv('data/pileaute/settler_data/all/TSS_eff_all.csv', index_col=0)
df.index=pd.to_datetime(df.index)

image_dates=os.listdir('data/pileaute/beluchting/microscopic_images_pileaute')
image_dates=pd.to_datetime(image_dates).sort_values()
all_dates = df.index.union(image_dates)

df_interpol=df.copy()
df_interpol=df_interpol.reindex(all_dates)
df_interpol = df_interpol.interpolate(method='time')

plt.figure(figsize=(12,3))
plt.plot(df_interpol, '.-', label='interpolated')
plt.plot(df, '.-', label='original')
plt.yscale('log')
plt.legend()


df_interpol.to_csv('data/pileaute/settler_data/all/TSS_eff_all_interpolated.csv', index=True)