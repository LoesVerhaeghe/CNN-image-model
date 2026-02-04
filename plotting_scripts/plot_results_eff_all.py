import pickle
import os
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##### watch out!! here in this script you still need to add mech model output to pred, also check this for other TSS res scripts

output_path='output/pileaute/settling/TSS_all'

all_labels = []
all_preds = []
all_dates=[]
for output in listdir(output_path):
    if output.endswith('.pk'):
        with open(os.path.join(output_path, output), 'rb') as f:
            data = pickle.load(f)
            labels = np.array(data[5]).flatten()
            preds = np.array(data[6]).flatten()
            dates = np.array([d[0] if isinstance(d, (list, tuple, np.ndarray)) else d for d in data[8]])

            all_labels.extend(labels)
            all_preds.extend(preds)
            all_dates.extend(dates)
df = pd.DataFrame({
    'date': all_dates,
    'label': all_labels,
    'pred': all_preds
})

df_avg = df.groupby('date', as_index=False).agg(
    label_mean=('label', 'mean'),
    pred_mean=('pred', 'mean'),
    pred_std=('pred', 'std')
)

df_avg=df_avg.set_index('date')
df_avg.index=pd.to_datetime(df_avg.index)


df=pd.read_excel('data/pileaute/settler_data/filtered_data/filtered_TSS_data.xlsx', sheet_name='TSS_effluent', index_col=0)
df.index=pd.to_datetime(df.index)


# --- Prepare data for plotting ---
y_true = df_avg['label_mean']
y_predicted = df_avg['pred_mean']
std_dev = df_avg['pred_std']

y_pred_upper = y_predicted + std_dev
y_pred_lower = y_predicted - std_dev

# #### make plot

# Define train and test indices
train_indices= list(range(0, 80))       
test_indices = list(range(80, 112))            
plt.rcParams.update({'font.size': 12})    

plt.figure(figsize=(14, 3), dpi=200)
plt.plot(y_true, '.-', label='Measurements', color='blue')
plt.plot(y_predicted.iloc[train_indices], '.-', label='Model predictions (train)', color='orange')
plt.plot(y_predicted.iloc[test_indices], '.-', label='Model predictions (test)', color='red')
plt.xlabel("Time")
plt.ylabel("TSS effluent (mg/L)")
plt.legend()
plt.yscale('log')
#plt.savefig('/home/loesv/all_results/pileaute/TSS_all/CNN.png',  bbox_inches='tight', dpi=250)
#plt.savefig('/home/loesv/all_results/pileaute/TSS_all/CNN.pdf',  bbox_inches='tight', dpi=250)
plt.show()


##################

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

plt.figure(figsize=(6, 6), dpi=250)
plt.rcParams.update({'font.size': 12})

# Extract true values and predictions for each split
y_true_train = y_true.iloc[train_indices]
y_pred_train = y_predicted.iloc[train_indices]
y_true_test = y_true.iloc[test_indices]
y_pred_test = y_predicted.iloc[test_indices]

# Scatter plots
plt.scatter(y_true_train, y_pred_train, color='orange', alpha=0.7, label='Train')
plt.scatter(y_true_test, y_pred_test, color='red', alpha=0.7, label='Test')

# 1:1 line
lims = [
    min(y_true.min(), y_predicted.min()),
    max(y_true.max(), y_predicted.max())
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
#plt.savefig('/home/loesv/all_results/pileaute/TSS_all/CNN_scatterplot.png',  bbox_inches='tight', dpi=250)
#plt.savefig('/home/loesv/all_results/pileaute/TSS_all/CNN_scatterplot.pdf',  bbox_inches='tight', dpi=250)
plt.show()
