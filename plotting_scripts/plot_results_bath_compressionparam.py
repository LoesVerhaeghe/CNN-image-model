import pickle
import os
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#output_path='output/bath/compression_params'
output_path='output/bath/compression_params_withDNAinput'

all_labels = []
all_preds = []
all_dates=[]
for output in listdir(output_path):
    if output.endswith('.pk'):
        with open(os.path.join(output_path, output), 'rb') as f:
            data = pickle.load(f)
            labels = np.array(data[5])
            preds = np.array(data[6])
            dates = np.array([d[0] if isinstance(d, (list, tuple, np.ndarray)) else d for d in data[8]])

            all_labels.extend(labels)
            all_preds.extend(preds)
            all_dates.extend(dates)
df = pd.DataFrame({
    'date': all_dates,
    'label_Xinfi': [x[0] for x in all_labels],
    'label_Vc': [x[1] for x in all_labels],
    'label_rc': [x[2] for x in all_labels],
    'pred_Xinfi': [x[0] for x in all_preds],
    'pred_Vc': [x[1] for x in all_preds],
    'pred_rc': [x[2] for x in all_preds],
})

df_avg = df.groupby('date', as_index=False).agg(
    label_Xinfi_mean=('label_Xinfi', 'mean'),
    label_Vc_mean=('label_Vc', 'mean'),
    label_rc_mean=('label_rc', 'mean'),
    pred_Xinfi_mean=('pred_Xinfi', 'mean'),
    pred_Vc_mean=('pred_Vc', 'mean'),
    pred_rc_mean=('pred_rc', 'mean'),
)

df_avg=df_avg.set_index('date')
df_avg.index=pd.to_datetime(df_avg.index)

# --- Prepare data for plotting ---
y_true1 = df_avg['label_Xinfi_mean']
y_predicted1 = df_avg['pred_Xinfi_mean']

# Define train and test indices
train_indices1 = list(range(0, 11))
train_indices2= list(range(16, 23))  # 0–12 and 18–23
train_indices=list(range(0, 11))+list(range(16, 23))
test_indices = list(range(11, 16))  # 13–17

#plot time series
plt.rcParams.update({'font.size': 11})    
plt.figure(figsize=(14, 3), dpi=200)
plt.plot(y_true1, '.-', label='Measurements', color='blue')
plt.plot(y_predicted1.iloc[train_indices1], '.-', label='Model predictions (train)', color='orange')
plt.plot(y_predicted1.iloc[train_indices2], '.-', color='orange')
plt.plot(y_predicted1.iloc[test_indices], '.-', label='Model predictions (test)', color='red')
plt.xlabel("Time")
plt.ylabel("Xinfi")
plt.legend()
plt.gcf().autofmt_xdate()
#plt.savefig('/home/loesv/all_results/bath/compression_params/CNN.png',  bbox_inches='tight', dpi=250)
#plt.savefig('/home/loesv/all_results/bath/compression_params/CNN.pdf',  bbox_inches='tight', dpi=250)
plt.show()


from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

plt.figure(figsize=(6, 6), dpi=250)
plt.rcParams.update({'font.size': 12})

# Extract true values and predictions for each split
y_true_train = y_true1.iloc[train_indices]
y_pred_train = y_predicted1.iloc[train_indices]
y_true_test = y_true1.iloc[test_indices]
y_pred_test = y_predicted1.iloc[test_indices]

# Scatter plots
plt.scatter(y_true_train, y_pred_train, color='orange', alpha=0.7, label='Train')
plt.scatter(y_true_test, y_pred_test, color='red', alpha=0.7, label='Test')

# 1:1 line
lims = [
    min(y_true1.min(), y_predicted1.min()),
    max(y_true1.max(), y_predicted1.max())
]
plt.plot(lims, lims, 'k--', alpha=0.8, label='1:1 line')

plt.xlabel("Measured")
plt.ylabel("Predicted")
plt.legend()

from scipy.stats import pearsonr, spearmanr

def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    return r2, rmse, mae, pearson_corr, spearman_corr

r2_train, rmse_train, mae_train, pearson_train, spearman_train = metrics(y_true_train, y_pred_train)
r2_test, rmse_test, mae_test, pearson_test, spearman_test = metrics(y_true_test, y_pred_test)

# Annotate metrics in the plot
textstr = '\n'.join((
    f"Train: R²={r2_train:.2f}, RMSE={rmse_train:.2f}, MAE={mae_train:.2f}, "
    f"Pearson={pearson_train:.2f}, Spearman={spearman_train:.2f}",
    f"Test:  R²={r2_test:.2f}, RMSE={rmse_test:.2f}, MAE={mae_test:.2f}, "
    f"Pearson={pearson_test:.2f}, Spearman={spearman_test:.2f}"
))

plt.figtext(0.5, -0.05, textstr, ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
plt.tight_layout()
#plt.savefig('/home/loesv/all_results/bath/compression_params/CNN_scatterplot.png',  bbox_inches='tight', dpi=250)
#plt.savefig('/home/loesv/all_results/bath/compression_params/CNN_scatterplot.pdf',  bbox_inches='tight', dpi=250)
plt.show()

# --- Prepare data for plotting ---
y_true2 = df_avg['label_Vc_mean']
y_predicted2 = df_avg['pred_Vc_mean']

# Define train and test indices
train_indices1 = list(range(0, 11))
train_indices2= list(range(16, 23))  # 0–12 and 18–23
train_indices=list(range(0, 11))+list(range(16, 23))
test_indices = list(range(11, 16))  # 13–17

#plot time series
plt.rcParams.update({'font.size': 11})    
plt.figure(figsize=(14, 3), dpi=200)
plt.plot(y_true2, '.-', label='Measurements', color='blue')
plt.plot(y_predicted2.iloc[train_indices1], '.-', label='Model predictions (train)', color='orange')
plt.plot(y_predicted2.iloc[train_indices2], '.-', color='orange')
plt.plot(y_predicted2.iloc[test_indices], '.-', label='Model predictions (test)', color='red')
plt.xlabel("Time")
plt.ylabel("Xinfi")
plt.legend()
plt.gcf().autofmt_xdate()
#plt.savefig('/home/loesv/all_results/bath/compression_params/CNN.png',  bbox_inches='tight', dpi=250)
#plt.savefig('/home/loesv/all_results/bath/compression_params/CNN.pdf',  bbox_inches='tight', dpi=250)
plt.show()


from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

plt.figure(figsize=(6, 6), dpi=250)
plt.rcParams.update({'font.size': 12})

# Extract true values and predictions for each split
y_true_train = y_true2.iloc[train_indices]
y_pred_train = y_predicted2.iloc[train_indices]
y_true_test = y_true2.iloc[test_indices]
y_pred_test = y_predicted2.iloc[test_indices]

# Scatter plots
plt.scatter(y_true_train, y_pred_train, color='orange', alpha=0.7, label='Train')
plt.scatter(y_true_test, y_pred_test, color='red', alpha=0.7, label='Test')

# 1:1 line
lims = [
    min(y_true2.min(), y_predicted2.min()),
    max(y_true2.max(), y_predicted2.max())
]
plt.plot(lims, lims, 'k--', alpha=0.8, label='1:1 line')

plt.xlabel("Measured")
plt.ylabel("Predicted")
plt.legend()

from scipy.stats import pearsonr, spearmanr

def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    return r2, rmse, mae, pearson_corr, spearman_corr

r2_train, rmse_train, mae_train, pearson_train, spearman_train = metrics(y_true_train, y_pred_train)
r2_test, rmse_test, mae_test, pearson_test, spearman_test = metrics(y_true_test, y_pred_test)

# Annotate metrics in the plot
textstr = '\n'.join((
    f"Train: R²={r2_train:.2f}, RMSE={rmse_train:.2f}, MAE={mae_train:.2f}, "
    f"Pearson={pearson_train:.2f}, Spearman={spearman_train:.2f}",
    f"Test:  R²={r2_test:.2f}, RMSE={rmse_test:.2f}, MAE={mae_test:.2f}, "
    f"Pearson={pearson_test:.2f}, Spearman={spearman_test:.2f}"
))

plt.figtext(0.5, -0.05, textstr, ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
plt.tight_layout()
#plt.savefig('/home/loesv/all_results/bath/compression_params/CNN_scatterplot.png',  bbox_inches='tight', dpi=250)
#plt.savefig('/home/loesv/all_results/bath/compression_params/CNN_scatterplot.pdf',  bbox_inches='tight', dpi=250)
plt.show()

# --- Prepare data for plotting ---
y_true3 = df_avg['label_rc_mean']
y_predicted3 = df_avg['pred_rc_mean']

# Define train and test indices
train_indices1 = list(range(0, 13))
train_indices2= list(range(18, 23))  # 0–12 and 18–23
train_indices=list(range(0, 13))+list(range(18, 23))
test_indices = list(range(13, 18))  # 13–17

#plot time series
plt.rcParams.update({'font.size': 11})    
plt.figure(figsize=(14, 3), dpi=200)
plt.plot(y_true3, '.-', label='Measurements', color='blue')
plt.plot(y_predicted3.iloc[train_indices1], '.-', label='Model predictions (train)', color='orange')
plt.plot(y_predicted3.iloc[train_indices2], '.-', color='orange')
plt.plot(y_predicted3.iloc[test_indices], '.-', label='Model predictions (test)', color='red')
plt.xlabel("Time")
plt.ylabel("Xinfi")
plt.legend()
plt.gcf().autofmt_xdate()
#plt.savefig('/home/loesv/all_results/bath/compression_params/CNN.png',  bbox_inches='tight', dpi=250)
#plt.savefig('/home/loesv/all_results/bath/compression_params/CNN.pdf',  bbox_inches='tight', dpi=250)
plt.show()


from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

plt.figure(figsize=(6, 6), dpi=250)
plt.rcParams.update({'font.size': 12})

# Extract true values and predictions for each split
y_true_train = y_true3.iloc[train_indices]
y_pred_train = y_predicted3.iloc[train_indices]
y_true_test = y_true3.iloc[test_indices]
y_pred_test = y_predicted3.iloc[test_indices]

# Scatter plots
plt.scatter(y_true_train, y_pred_train, color='orange', alpha=0.7, label='Train')
plt.scatter(y_true_test, y_pred_test, color='red', alpha=0.7, label='Test')

# 1:1 line
lims = [
    min(y_true3.min(), y_predicted3.min()),
    max(y_true3.max(), y_predicted3.max())
]
plt.plot(lims, lims, 'k--', alpha=0.8, label='1:1 line')

plt.xlabel("Measured")
plt.ylabel("Predicted")
plt.legend()

from scipy.stats import pearsonr, spearmanr

def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    return r2, rmse, mae, pearson_corr, spearman_corr

r2_train, rmse_train, mae_train, pearson_train, spearman_train = metrics(y_true_train, y_pred_train)
r2_test, rmse_test, mae_test, pearson_test, spearman_test = metrics(y_true_test, y_pred_test)

# Annotate metrics in the plot
textstr = '\n'.join((
    f"Train: R²={r2_train:.2f}, RMSE={rmse_train:.2f}, MAE={mae_train:.2f}, "
    f"Pearson={pearson_train:.2f}, Spearman={spearman_train:.2f}",
    f"Test:  R²={r2_test:.2f}, RMSE={rmse_test:.2f}, MAE={mae_test:.2f}, "
    f"Pearson={pearson_test:.2f}, Spearman={spearman_test:.2f}"
))

plt.figtext(0.5, -0.05, textstr, ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
plt.tight_layout()
#plt.savefig('/home/loesv/all_results/bath/compression_params/CNN_scatterplot.png',  bbox_inches='tight', dpi=250)
#plt.savefig('/home/loesv/all_results/bath/compression_params/CNN_scatterplot.pdf',  bbox_inches='tight', dpi=250)
plt.show()


############# plot all 3 together


plt.rcParams.update({'font.size': 11})    

fig, axes = plt.subplots(3, 1, figsize=(14, 8), dpi=200, sharex=True)

# -------------------
# Subplot 1: Xinfi
axes[0].plot(y_true1, '.-', label='Measurements Xinfi', color='blue')
axes[0].plot(y_predicted1.iloc[train_indices1], '.-', label='Model predictions (train)', color='orange')
axes[0].plot(y_predicted1.iloc[train_indices2], '.-', color='orange')
axes[0].plot(y_predicted1.iloc[test_indices], '.-', label='Model predictions (test)', color='red')
axes[0].set_ylabel("Xinfi")
axes[0].legend()

# -------------------
# Subplot 2: Vc
axes[1].plot(y_true2, '.-', label='Measurements Vc', color='blue')
axes[1].plot(y_predicted2.iloc[train_indices1], '.-', label='Model predictions (train)', color='orange')
axes[1].plot(y_predicted2.iloc[train_indices2], '.-', color='orange')
axes[1].plot(y_predicted2.iloc[test_indices], '.-', label='Model predictions (test)', color='red')
axes[1].set_ylabel("Vc")
axes[1].legend()

# -------------------
# Subplot 3: rC
axes[2].plot(y_true3, '.-', label='Measurements rC', color='blue')
axes[2].plot(y_predicted3.iloc[train_indices1], '.-', label='Model predictions (train)', color='orange')
axes[2].plot(y_predicted3.iloc[train_indices2], '.-', color='orange')
axes[2].plot(y_predicted3.iloc[test_indices], '.-', label='Model predictions (test)', color='red')
axes[2].set_ylabel("rC")
axes[2].set_xlabel("Time")
axes[2].legend()

plt.gcf().autofmt_xdate()
plt.tight_layout()
#plt.savefig('/home/loesv/all_results/bath/compression_params_images/CNN.png',  bbox_inches='tight', dpi=250)
#plt.savefig('/home/loesv/all_results/bath/compression_params_images/CNN.pdf',  bbox_inches='tight', dpi=250)
plt.savefig('/home/loesv/all_results/bath/compression_params_imagesandDNA/CNN.png',  bbox_inches='tight', dpi=250)
plt.savefig('/home/loesv/all_results/bath/compression_params_imagesandDNA/CNN.pdf',  bbox_inches='tight', dpi=250)
plt.show()
