import os
import pandas as pd
from os import listdir
from datetime import datetime

svi_csv_path = "data/SVI.csv"  # Original SVI data with dates
image_root = "data/images_pileaute"  # Root folder containing dated image subfolders

svi_df = pd.read_csv(svi_csv_path, index_col=[0])

# Ensure date column is datetime
svi_df.index = pd.to_datetime(svi_df.index)

image_dates = listdir(image_root)
image_dates = sorted(set(image_dates))
image_dates = pd.to_datetime(image_dates)

# Interpolate SVI to those image dates 
image_dates_df = pd.DataFrame(index=image_dates)
interpolated = svi_df.reindex(svi_df.index.union(image_dates_df.index)).sort_index()
interpolated = interpolated.interpolate(method='time')  # time-based linear interpolation

svi_interp_df = interpolated.loc[image_dates]

# === Step 4: Save to new CSV ===
svi_interp_df.to_csv("data/SVI_interpolated.csv")

import matplotlib.pyplot as plt
original = pd.read_csv("data/SVI.csv", index_col=[0])
interpolated = pd.read_csv("data/SVI_interpolated.csv", index_col=[0])
original.index = pd.to_datetime(original.index)
interpolated.index = pd.to_datetime(interpolated.index)

plt.figure(figsize=(10, 5))
plt.plot(original, 'o-', label='Original SVI', color='blue')
plt.plot(interpolated, 'x--', label='Interpolated SVI', color='orange')

plt.xlabel("Date")
plt.ylabel("SVI")
plt.legend()
plt.show()