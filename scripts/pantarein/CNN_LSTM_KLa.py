import pickle
import os
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir, path as os_path 
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.optim as optim
from copy import deepcopy

torch.set_num_threads(4)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # for single-GPU

    # For deterministic behavior (might slow things down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# data aggregeren in matrices (3, 24, 1028), aanvullen met 0 als er geen 20 images per dag zijn
class DailySequenceDataset(Dataset):
    def __init__(self, features_per_day, target_per_day, seq_len=3, n_images=24):
        '''
        features_per_day: dict of date -> list of feature arrays per image
        target_per_day: dict of date -> targetvalue
        '''
        self.seq_len = seq_len
        self.n_images = n_images

        self.dates = sorted(list(set(features_per_day.keys()) & set(target_per_day.keys()))) #only use dates that have both features and a target
        self.features_per_day = features_per_day
        self.targets_per_day = target_per_day

        # Maak geldige sequenties
        self.samples = self.create_sequences()

    def create_sequences(self):
        samples = []

        for i in range(self.seq_len - 1, len(self.dates)):
            sequence_dates = self.dates[i - self.seq_len + 1:i+1] # if i => forecasting. if i+1 => prediction
            feature_seq = []
            
            for date in sequence_dates:
                features = self.features_per_day[date]
                n_features=len(features)
                if n_features >= self.n_images:
                    selected=features[:self.n_images] # Selecteer precies n_images (bijv. eerste of random)
                else:
                    selected=features+[np.zeros(self.features_per_day[date][0].shape)] * (self.n_images-n_features)
                feature_seq.append(np.stack(selected))

            feature_tensor = np.stack(feature_seq)  # shape: (seq_len, n_images, feature_dim)
            target = self.targets_per_day[self.dates[i]]
            samples.append((feature_tensor, target))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature_seq, target = self.samples[idx]
        feature_seq = torch.tensor(feature_seq, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        return feature_seq, target

class TransformerDailyAggregator(nn.Module):
    def __init__(self, input_dim=640, embed_dim=512, n_heads=4, n_layers=2, output_dim=2048):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_projector = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x: (B, N, input_dim) where N = n_images_per_day
        """
        x = self.embedding(x)  # (B, N, embed_dim)
        x = self.transformer(x)  # (B, N, embed_dim)
        x = x.mean(dim=1)  # mean-pool over N images
        return self.output_projector(x)  # (B, output_dim)
    
# 3. LSTM 
class LstmArchitecture(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super(LstmArchitecture, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)               # out: (batch, seq_len, hidden)
        out = out[:, -1, :]                 # Take output at last time step
        return self.fc(out).squeeze()       # Final regression prediction

class FullPipelineModel(nn.Module):
    def __init__(self, 
                 input_dim=640, 
                 embed_dim=512, 
                 n_heads=4, 
                 n_layers=2, 
                 agg_output_dim=2048, 
                 lstm_hidden=32, 
                 num_layers=1):
        super().__init__()

        # Aggregates 24 images x 1028 features → 1 vector
        self.daily_aggregator = TransformerDailyAggregator(
            input_dim=input_dim,
            embed_dim=embed_dim,
            n_heads=n_heads, 
            n_layers=n_layers,
            output_dim=agg_output_dim
        )

        # LSTM model
        self.lstm_model = LstmArchitecture(
            input_size=agg_output_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers
        )

    def forward(self, x):
        # x: [batch, seq_len=3, n_images=24, feature_dim=1028]
        B, S, N, D = x.shape

        # Flatten to [B*S, N, D] to put into daily aggregator
        x = x.view(B * S, N, D)

        # Aggregate each day → [B*S, 2048]
        aggregated = self.daily_aggregator(x)

        # Reshape back to sequence → [B, S, 2048]
        aggregated_seq = aggregated.view(B, S, -1)
        # Pass through your LSTM
        return self.lstm_model(aggregated_seq)

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs=20):
    avg_train_losses = []
    avg_val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        # train loop
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).float()
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_losses.append(avg_train_loss)

        # val loop (only if val_loader is given)
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device).float()
                    preds = model(X)
                    loss = criterion(preds, y)
                    val_loss += loss.item() * X.size(0)

            avg_val_loss = val_loss / len(val_loader.dataset)
            avg_val_losses.append(avg_val_loss)

            if scheduler:
                scheduler.step(avg_val_loss)

            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

            # save best model (based on val)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = deepcopy(model.state_dict())
        else:
            # no validation → just print training loss
            if scheduler:
                scheduler.step(avg_train_loss)

            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

    # restore best model (if we tracked one)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    #plot the losses
    plt.figure()
    plt.plot(avg_train_losses, label='train losses')
    plt.plot(avg_val_losses, label='val losses')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.legend()

    return model, best_val_loss if val_loader is not None else avg_train_loss

#####################################################################################################################################################################
# Load ALL Features, Aggregate, Create Labels and Folder Mapping 

output_path='output/pantarein/kla'
images_base_folder = 'data/pantarein/image_data_pantarein_structured'
df=pd.read_csv('data/pantarein/kla.csv', index_col=0)

all_image_folders = sorted(listdir(images_base_folder))
num_folders_total = len(all_image_folders)

all_img_features = []
for output in listdir(output_path):
    if output.endswith('.pk'):
        with open(os.path.join(output_path, output), 'rb') as f:
            data = pickle.load(f)
            img_features=data[7]
            all_img_features.append(img_features)

all_fold_pred=[]
for fold_im_features in all_img_features:      
    features_per_date={}
    target_per_date={}
    total_img_count = 0
    all_dates=[]
    for folder in all_image_folders:
        if folder not in df.index:
            continue
        if folder not in target_per_date:
            target_per_date[folder]=[]
        # Path to embeddings for this folder
        path_to_folder = f"{images_base_folder}/{folder}"
        images_in_folder_count = 0
        images_list_embeddings = listdir(path_to_folder)
        for image_file in images_list_embeddings:
            if folder not in features_per_date:
                features_per_date[folder]=[]
            try:
                # Save features
                img_path = f"{path_to_folder}/{image_file}"
                embedding = fold_im_features[total_img_count]
                features_per_date[folder].append(embedding)
                total_img_count += 1
                images_in_folder_count += 1
            except Exception as e:
                print(f"Error loading or processing {img_path}: {e}")
        if images_in_folder_count > 0:
            target_per_date[folder]=df['KLa'].loc[folder].item() # Save label per date
        all_dates.append(folder)
        
    seq_len=2
    #### test dataset class
    dataset = DailySequenceDataset(features_per_date, target_per_date, seq_len=seq_len, n_images=5)

    # Check how many sequences were created
    print(f"Number of sequences: {len(dataset.samples)}")
    # Check first sequence shapes and target
    for i, (features, target) in enumerate(dataset.samples):
        print(f"Sequence {i} feature shape: {features.shape}")  # should be (seq_len, n_images, feature_dim)
        print(f"Sequence {i} target: {target}")

    # --- Split Based on Time (Processed Folders) ---
    # Use the number of *processed* folders for splitting
    train_indices_folders=np.arange(0, 70)
    val_indices_folders = np.arange(70, 78) 
    test_indices_folders=np.arange(78, 113)

    train_set = Subset(dataset, train_indices_folders)
    val_set = Subset(dataset, val_indices_folders)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=True)

    ## define and train model
    torch.cuda.set_device(0) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_temp = FullPipelineModel(input_dim=640, 
                          #transformer parameters
                          embed_dim=128,
                          n_heads=8, 
                          n_layers=2,
                          agg_output_dim=128,
                          #lstm parameters
                          lstm_hidden=32,
                          num_layers=1).to(device)
    optimizer = optim.Adam(model_temp.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    #first do this to estimate how many epochs needed
    #_, _ = train_model(model_temp, train_loader, val_loader, optimizer, None, criterion, epochs=500)

    #then train your model using all data
    model = FullPipelineModel(input_dim=640, 
                          #transformer parameters
                          embed_dim=128,
                          n_heads=8, 
                          n_layers=2,
                          agg_output_dim=512,
                          #lstm parameters
                          lstm_hidden=32,
                          num_layers=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()
    train_val_indices_folders=np.arange(0, 78)
    train_val_set = Subset(dataset, train_val_indices_folders)
    train_val_loader = DataLoader(train_val_set, batch_size=16, shuffle=True)
    trained_model, best_val_loss=train_model(model, train_val_loader, None, optimizer, None, criterion, epochs=250)


    ### testing
    full_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    all_preds=[]
    trained_model.eval()
    with torch.no_grad():
        for X, y in full_loader:
            X, y = X.to(device), y.to(device).float()
            preds = trained_model(X)
            all_preds.append(preds.cpu().item())
    all_fold_pred.append(all_preds)

    all_dates=pd.to_datetime(all_dates)
    df.index=pd.to_datetime(df.index)

    # --- Construct Model preds ---
    predictions = pd.Series(
        all_preds,
        index=all_dates[seq_len-1:]
    )

    # --- Plotting Results ---
    plt.figure(figsize=(14, 3), dpi=200)
    plt.rcParams.update({'font.size': 12})    
    plt.plot(df['KLa'][all_dates].iloc[seq_len-1:], '.-', label='Measurements', color='blue')
    plt.plot(predictions.iloc[0:len(train_indices_folders)], '.-', label='predictions (train)', color='orange')
    plt.plot(predictions.iloc[len(train_indices_folders):len(train_indices_folders)+len(val_indices_folders)], '.-', label='predictions (val)', color='green')
    plt.plot(predictions.iloc[len(train_indices_folders)+len(val_indices_folders):], '.-', label='predictions (test)', color='red')
    plt.xlabel("Time")
    plt.ylabel("KLa")
    plt.legend()
    plt.show()

avg_preds = np.mean(all_fold_pred, axis=0)
std_dev = np.std(all_fold_pred, axis=0)

# --- Construct Model preds ---
predictions = pd.Series(
    avg_preds,
    index=all_dates[seq_len-1:]
)

# --- Plotting Results ---
plt.figure(figsize=(14, 3), dpi=200)
plt.rcParams.update({'font.size': 12})    
plt.plot(df['KLa'][all_dates].iloc[seq_len-1:], '.-', label='Measurements', color='blue')
plt.plot(predictions.iloc[0:len(train_indices_folders)+len(val_indices_folders)], '.-', label='Model predictions (train)', color='orange')
plt.plot(predictions.iloc[len(train_indices_folders)+len(val_indices_folders):], '.-', label='Model predictions (test)', color='red')
plt.xlabel("Time")
plt.ylabel("KLa (1/h)")
plt.legend()
plt.savefig('/home/loesv/all_results/pantarein/KLa/CNN_LSTM.png', dpi=250)
plt.savefig('/home/loesv/all_results/pantarein/KLa/CNN_LSTM.pdf', dpi=250)
plt.show()

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

plt.figure(figsize=(6, 6), dpi=250)
plt.rcParams.update({'font.size': 12})

# Extract true values and predictions for each split
y_true_train = df['KLa'][all_dates].iloc[seq_len-1:].iloc[0:len(train_indices_folders)+len(val_indices_folders)]
y_pred_train = predictions.iloc[0:len(train_indices_folders)+len(val_indices_folders)]
y_true_test = df['KLa'][all_dates].iloc[seq_len-1:].iloc[len(train_indices_folders)+len(val_indices_folders):]
y_pred_test = predictions.iloc[len(train_indices_folders)+len(val_indices_folders):]

# Scatter plots
plt.scatter(y_true_train, y_pred_train, color='orange', alpha=0.7, label='Train')
plt.scatter(y_true_test, y_pred_test, color='red', alpha=0.7, label='Test')

# 1:1 line
lims = [
    min(df['KLa'][all_dates].iloc[seq_len-1:].min(), predictions.min()),
        max(df['KLa'][all_dates].iloc[seq_len-1:].max(), predictions.max())
        ]
plt.plot(lims, lims, 'k--', alpha=0.8, label='1:1 line')

plt.xlabel("Measured")
plt.ylabel("Predicted")
plt.legend()


from scipy.stats import pearsonr, spearmanr

def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    return r2, mse, mae, pearson_corr, spearman_corr

r2_train, mse_train, mae_train, pearson_train, spearman_train = metrics(y_true_train, y_pred_train)
r2_test, mse_test, mae_test, pearson_test, spearman_test = metrics(y_true_test, y_pred_test)

# Annotate metrics in the plot
textstr = '\n'.join((
    f"Train: R²={r2_train:.3g}, MSE={mse_train:.3g}, MAE={mae_train:.3g}, "
    f"Pearson={pearson_train:.3g}, Spearman={spearman_train:.3g}",
    f"Test:  R²={r2_test:.3g}, MSE={mse_test:.3g}, MAE={mae_test:.3g}, "
    f"Pearson={pearson_test:.3g}, Spearman={spearman_test:.3g}"
))

plt.figtext(0.5, -0.05, textstr, ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
plt.tight_layout()
plt.savefig('/home/loesv/all_results/pantarein/KLa/CNN_LSTM_scatterplot.png',  bbox_inches='tight', dpi=250)
plt.savefig('/home/loesv/all_results/pantarein/KLa/CNN_LSTM_scatterplot.pdf',  bbox_inches='tight', dpi=250)
plt.show()
