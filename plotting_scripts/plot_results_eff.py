import pickle
import os
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_path='output/settling/10fold/output_TSSeff_error_train1'
images_base_folder = 'data/settler_data/filtered_data/images_pileaute_settling_filtered'

image_folders = sorted(listdir(images_base_folder)) 

all_labels = []
all_preds = []
for output in listdir(output_path):
    if output.endswith('.pk'):
        with open(os.path.join(output_path, output), 'rb') as f:
            data = pickle.load(f)
            labels=data[5]
            preds=data[6]
            all_labels.append(labels)
            all_preds.append(preds)

#calc average per fold
avg_preds = np.mean([all_preds[i] for i in range(10)], axis=0)
avg_labels= np.mean([all_labels[i] for i in range(10)], axis=0)

y_predicted = []
y_true = []
std_dev = []
i=0
for folder in image_folders:
    path = f"{images_base_folder}/{folder}/basin5/10x"
    images_list = listdir(path)
    temporary_pred=[]
    temporary_label=[]
    for image in images_list:
        pred = avg_preds[i]
        label=avg_labels[i]
        i+=1
        temporary_pred.append(pred)
        temporary_label.append(label)
    y_predicted.append(sum(temporary_pred)/len(temporary_pred))
    y_true.append(sum(temporary_label)/len(temporary_label))
    std_dev.append(np.std(temporary_pred))
y_predicted=np.array(y_predicted).reshape(-1)
y_true=np.array(y_true).reshape(-1)
all_image_folders_datetime=pd.to_datetime(image_folders)
df=pd.read_excel('data/settler_data/filtered_data/filtered_TSS_data.xlsx', sheet_name='TSS_effluent', index_col=0)
df.index=pd.to_datetime(df.index)

# --- Construct Model preds and Uncertainty Bands ---
# Ensure index alignment - crucial if folders were skipped
y_predicted = pd.Series(
    y_predicted+df['MechModelOutput'].values,
    index=all_image_folders_datetime
)

y_true=pd.Series(
    y_true+df['MechModelOutput'].values,
    index=all_image_folders_datetime
)

y_pred_upper = pd.Series(
    y_predicted.values + std_dev,
    index=all_image_folders_datetime
)
y_pred_lower = pd.Series(
    y_predicted.values - std_dev,
    index=all_image_folders_datetime
)

# #### make plot

# # TRAIN 1
# Define train and test indices
train_indices= list(range(0, 55))       
test_indices = list(range(55, 81))            
plt.rcParams.update({'font.size': 12})    

plt.figure(figsize=(14, 3), dpi=200)
plt.plot(y_true, '.-', label='Measurements', color='blue')

# Plot model predictions
plt.plot(y_predicted.iloc[train_indices], '.-', label='HM predictions (train)', color='orange')
plt.plot(y_predicted.iloc[test_indices], '.-', label='HM predictions (test)', color='red')
plt.plot(df['MechModelOutput'],'.-', label='Mechanistic model output', color='green')

# Plot Std Dev Band for Train – Part 1
plt.fill_between(y_predicted.index[train_indices],
                 y_pred_lower[train_indices],
                 y_pred_upper[train_indices],
                 color='orange', alpha=0.2, zorder=1)

# Plot Std Dev Band for Test
plt.fill_between(y_predicted.index[test_indices],
                 y_pred_lower[test_indices],
                 y_pred_upper[test_indices],
                 color='red', alpha=0.2, zorder=1)

plt.xlabel("Time")
plt.ylabel("TSS effluent (mg/L)")
plt.legend()
plt.show()


# ##TRAIN 2
# ##Define train and test indices
# train_indices_part1 = list(range(0, 27))      
# train_indices_part2 = list(range(55, 81))      
# test_indices = list(range(27, 55))            

# plt.figure(figsize=(14, 3), dpi=150)
# plt.plot(y_true, '.-', label='Measurements', color='blue')

# # Plot model predictions
# plt.plot(y_predicted.iloc[train_indices_part1], '.-', label='Model predictions (train)', color='orange')
# plt.plot(y_predicted.iloc[train_indices_part2], '.-', color='orange')
# plt.plot(y_predicted.iloc[test_indices], '.-', label='Model predictions (test)', color='red')

# # Plot Std Dev Band for Train – Part 1
# plt.fill_between(y_predicted.index[train_indices_part1],
#                  y_pred_lower[train_indices_part1],
#                  y_pred_upper[train_indices_part1],
#                  color='orange', alpha=0.2, zorder=1)

# # Plot Std Dev Band for Train – Part 2
# plt.fill_between(y_predicted.index[train_indices_part2],
#                  y_pred_lower[train_indices_part2],
#                  y_pred_upper[train_indices_part2],
#                  color='orange', alpha=0.2, zorder=1)

# # Plot Std Dev Band for Test
# plt.fill_between(y_predicted.index[test_indices],
#                  y_pred_lower[test_indices],
#                  y_pred_upper[test_indices],
#                  color='red', alpha=0.2, zorder=1)

# plt.xlabel("Time")
# plt.ylabel("TSS effluent")
# plt.legend()
# plt.show()


# TRAIN 3
# Define train and test indices
# train_indices = list(range(28, 81))      
# test_indices = list(range(0, 28))            

# plt.figure(figsize=(14, 3), dpi=150)
# plt.plot(y_true, '.-', label='Measurements', color='blue')

# # Plot model predictions
# plt.plot(y_predicted.iloc[train_indices], '.-', label='Model predictions (train)', color='orange')
# plt.plot(y_predicted.iloc[test_indices], '.-', label='Model predictions (test)', color='red')

# # Plot Std Dev Band for Train – Part 1
# plt.fill_between(y_predicted.index[train_indices],
#                  y_pred_lower[train_indices],
#                  y_pred_upper[train_indices],
#                  color='orange', alpha=0.2, zorder=1)

# # Plot Std Dev Band for Test
# plt.fill_between(y_predicted.index[test_indices],
#                  y_pred_lower[test_indices],
#                  y_pred_upper[test_indices],
#                  color='red', alpha=0.2, zorder=1)

# plt.xlabel("Time")
# plt.ylabel("TSS effluent")
# plt.legend()
# plt.show()
