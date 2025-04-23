import pickle
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open('results/all_folds_predictions.pkl', 'rb') as f:
    data = pickle.load(f)

predictions=data['fold_predictions']
avg = np.mean([predictions[i] for i in range(10)], axis=0)

### calc average label per sample date
base_folder = 'data/images_pileaute'
image_folders = listdir(base_folder) 

predicted_SVI = []
std_dev = []
i=0
for folder in image_folders:
    path = f"{base_folder}/{folder}/basin5/10x"
    images_list = listdir(path)
    temporary=[]
    for image in images_list:
        pred = avg[i]
        i+=1
        temporary.append(pred)
    predicted_SVI.append(sum(temporary)/len(temporary))
    std_dev.append(np.std(temporary))

std_dev = np.array(std_dev)
predicted_SVI = np.array(predicted_SVI)

SVI_pred_upper = predicted_SVI + std_dev
SVI_pred_lower = predicted_SVI - std_dev

true_SVI=pd.read_csv('data/SVI_interpolated.csv', index_col=0)
true_SVI_uninterpolated=pd.read_csv('data/SVI.csv', index_col=0)
true_SVI_uninterpolated.index=pd.to_datetime(true_SVI_uninterpolated.index)
true_SVI.index=pd.to_datetime(true_SVI.index)
predicted_SVI = pd.DataFrame(predicted_SVI, index=true_SVI.index)

######################## plot
plot_split_index =round(65)

plt.figure(figsize=(14, 3), dpi=150)
plt.plot(true_SVI_uninterpolated['SVI'], '.-', label='Measurements', color='blue')
plt.plot(predicted_SVI.iloc[:plot_split_index], '.-', label='Model predictions (train)', color='orange')
plt.plot(predicted_SVI.iloc[plot_split_index:], '.-', label='Model predictions (test)', color='red')

# Plot Standard Deviation Band (Train) - use iloc
plt.fill_between(predicted_SVI.index[:plot_split_index],
                 SVI_pred_lower[:plot_split_index],
                 SVI_pred_upper[:plot_split_index],
                 color='orange', alpha=0.2, zorder=1)

# Plot Standard Deviation Band (Test) - use iloc
plt.fill_between(predicted_SVI.index[plot_split_index:],
                 SVI_pred_lower[plot_split_index:],
                 SVI_pred_upper[plot_split_index:],
                 color='red', alpha=0.2, zorder=1)

plt.xlabel("Time")
plt.ylabel("SVI (mL/g)")
plt.title("SVI (CNN convnext_nano)")
plt.yscale("log")  # Make y-axis logarithmic
plt.legend()
plt.show()