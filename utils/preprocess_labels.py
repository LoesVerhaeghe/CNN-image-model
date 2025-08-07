import os
import pandas as pd
from os import listdir
from datetime import datetime

#################### preprocess SVI

svi_csv_path = "data/SVI/cleaned_SVIs.csv"  # Original SVI data with dates
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
new_path="data/SVI/cleaned_SVIs_interpolated.csv"
svi_interp_df.to_csv(new_path)

import matplotlib.pyplot as plt
original = pd.read_csv(svi_csv_path, index_col=[0])
interpolated = pd.read_csv(new_path, index_col=[0])
original.index = pd.to_datetime(original.index)
interpolated.index = pd.to_datetime(interpolated.index)

plt.figure(figsize=(10, 5))
plt.plot(original, 'o-', label='Original SVI', color='blue')
plt.plot(interpolated, 'o-', label='Interpolated SVI', color='orange')

plt.xlabel("Date")
plt.ylabel("SVI")
plt.legend()
plt.show()

#################### preprocess KLa


kla_csv_path = "data/KLa/kla_values.csv"  # Original SVI data with dates
image_root = "data/images_pileaute"  # Root folder containing dated image subfolders

kla_df = pd.read_csv(kla_csv_path, index_col=[0])

# Ensure date column is datetime
kla_df.index = pd.to_datetime(kla_df.index)

image_dates = listdir(image_root)
image_dates = sorted(set(image_dates))
image_dates = pd.to_datetime(image_dates)

# Interpolate SVI to those image dates 
image_dates_df = pd.DataFrame(index=image_dates)
interpolated = kla_df.reindex(kla_df.index.union(image_dates_df.index)).sort_index()
interpolated = interpolated.interpolate(method='time')  # time-based linear interpolation

kla_interp_df = interpolated.loc[image_dates]

# === Step 4: Save to new CSV ===
kla_interp_df.to_csv("data/KLa/KLa_interpolated.csv")

import matplotlib.pyplot as plt
original = pd.read_csv("data/KLa/kla_values.csv", index_col=[0])
interpolated = pd.read_csv("data/KLa/KLa_interpolated.csv", index_col=[0])
original.index = pd.to_datetime(original.index)
interpolated.index = pd.to_datetime(interpolated.index)

plt.figure(figsize=(10, 5))
plt.plot(original, 'o-', label='Original KLa', color='blue')
plt.plot(interpolated, 'o-', label='Interpolated KLa', color='orange')

plt.xlabel("Date")
plt.ylabel("KLa")
plt.legend()
plt.show()


#################### preprocess extra input


COD_csv_path = "data/CODt_labmeasurement.csv"  # Original SVI data with dates
image_root = "data/images_pileaute"  # Root folder containing dated image subfolders

cod_df = pd.read_csv(COD_csv_path, index_col=[0])

# Ensure date column is datetime
cod_df.index = pd.to_datetime(cod_df.index)

image_dates = listdir(image_root)
image_dates = sorted(set(image_dates))
image_dates = pd.to_datetime(image_dates)

# Interpolate SVI to those image dates 
image_dates_df = pd.DataFrame(index=image_dates)
interpolated = cod_df.reindex(cod_df.index.union(image_dates_df.index)).sort_index()
interpolated = interpolated.interpolate(method='time')  # time-based linear interpolation

cod_interp_df = interpolated.loc[image_dates]

# === Step 4: Save to new CSV ===
cod_interp_df.to_csv("data/CODt_interpolated.csv")

import matplotlib.pyplot as plt
original = pd.read_csv("data/CODt_labmeasurement.csv", index_col=[0])
interpolated = pd.read_csv("data/CODt_interpolated.csv", index_col=[0])
original.index = pd.to_datetime(original.index)
interpolated.index = pd.to_datetime(interpolated.index)

plt.figure(figsize=(10, 5))
plt.plot(original, 'o-', label='Original CODt', color='blue')
plt.plot(interpolated, 'o-', label='Interpolated CODt', color='orange')

plt.xlabel("Date")
plt.ylabel("COD")
plt.legend()
plt.show()



#################### preprocess TSS error


TSS_eff_error_csv_path = "data/TSS_errors/TSS_eff_error.csv"  
image_root = "data/images_pileaute"  # Root folder containing dated image subfolders

TSS_eff_error_df = pd.read_csv(TSS_eff_error_csv_path, index_col=[0])

# Ensure date column is datetime
TSS_eff_error_df.index = pd.to_datetime(TSS_eff_error_df.index)

image_dates = listdir(image_root)
image_dates = sorted(set(image_dates))
image_dates = pd.to_datetime(image_dates)

# Interpolate SVI to those image dates 
image_dates_df = pd.DataFrame(index=image_dates)
interpolated = TSS_eff_error_df.reindex(TSS_eff_error_df.index.union(image_dates_df.index)).sort_index()
interpolated = interpolated.interpolate(method='time')  # time-based linear interpolation

TSS_eff_error_interp_df = interpolated.loc[image_dates]

# === Step 4: Save to new CSV ===
TSS_eff_error_interp_df.to_csv("data/TSS_errors/TSS_eff_error_interpolated.csv")

import matplotlib.pyplot as plt
original = pd.read_csv(TSS_eff_error_csv_path, index_col=[0])
interpolated = pd.read_csv("data/TSS_errors/TSS_eff_error_interpolated.csv", index_col=[0])
original.index = pd.to_datetime(original.index)
interpolated.index = pd.to_datetime(interpolated.index)

plt.figure(figsize=(10, 5))

plt.plot(interpolated, 'o-', label='Interpolated', color='orange')
plt.plot(original, 'o-', label='Original', color='blue')

plt.xlabel("Date")
plt.legend()
plt.show()