# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:00:00 2023

@author: Loes V (Adapted from scabini's script)

Script to generate predictions for the *entire* dataset using *each*
model trained during K-fold cross-validation.
"""

import argparse
import os
import numpy as np
import pickle
import time
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
import sklearn.model_selection
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix

from src import dataset
if torch.cuda.is_available():
    _ = torch.cuda.current_device()

FILENAME_BATCH_SIZE = 32 # <<< MAKE SURE THIS MATCHES the BATCH_SIZE in the saved filenames

def parse_args():
    parser = argparse.ArgumentParser(description="Tester: Generate predictions for ALL data using EACH K-Fold model")
    # --- Data, Paths ---
    parser.add_argument('--imagedatapath', type=str, default='data/images_pileaute', help='Path to load the image dataset')
    parser.add_argument('--labelpath', type=str, default='data/SVI_interpolated.csv', help='Path to load the labels')
    parser.add_argument('--output_path', type=str, default='output', help='Path WHERE MODELS WERE SAVED during training')
    parser.add_argument('--results_file', type=str, default='results/all_folds_predictions.pkl', help='File path to save predictions from all folds')

    # --- Model & Training Config (MUST MATCH TRAINING) ---
    parser.add_argument('--backbone', type=str, default='convnext_nano', help='Pretrained model name (must match trained model)')
    parser.add_argument('--target', type=str, default='SVI', help='Target variable trained on (e.g., SVI, bulking)')
    parser.add_argument('--tl', action='store_true', default=True, help='Set if transfer learning was used during training (for filename)')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate used during training (for filename)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs model was trained for (for filename)')

    # --- K-Fold Config (MUST MATCH TRAINING) ---
    parser.add_argument('--K', type=int, default=10, help='K-fold splits used during training')
    parser.add_argument('--seed', type=int, default=666, help='Base random seed used during training for K-fold split (needed for TARGET_SCALE)')

    # --- Testing Specific ---
    parser.add_argument('--gpu', type=str, default='1', help='GPU ID to use for testing (see nvidia-smi)')
    parser.add_argument('--test_batch_size', type=int, default=64, help='Batch size for testing inference')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for data loading')

    return parser.parse_args()

def get_model_path(args, fold):
    """Constructs the path to the saved model file for a specific fold."""
    subfolders = args.output_path + '/'
    base_filename = f"{args.seed}_{args.K}_{fold}_{args.backbone}_{args.lr}_{args.epochs}_batch{FILENAME_BATCH_SIZE}_{args.target}"
    if args.tl:
        mode = 'TRANSFERLEARNING'
    else:
        mode = 'FROMSCRATCH'
    model_filename = f"{base_filename}_{mode}.pk_NETWORK.pt" # Match the exact extension used
    return os.path.join(subfolders, model_filename)

if __name__ == "__main__":
    args = parse_args()

    # --- Device Setup ---
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {args.gpu} (mapped to {device})")
    else:
        device = "cpu"
        print("CUDA not available, using CPU.")

    # --- Image Transformations (MUST match validation transform from training) ---
    imgdimm = (384, 512) # <<< MAKE SURE THIS MATCHES the imgdimm used during training
    averages = (0.485, 0.456, 0.406)
    variances = (0.229, 0.224, 0.225)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(imgdimm),
        transforms.Normalize(averages, variances),
    ])

    # --- Load Full Dataset ---
    print(f"Loading dataset from: {args.imagedatapath} with labels: {args.labelpath}")
    full_dataset = dataset.MicroscopicImages(
        root=args.imagedatapath,
        magnification=10, # Assuming magnification=10, adjust if needed
        label_path=args.labelpath,
        image_type='all',
        transform=val_transform # Use validation transform for testing
    )
    dataset_size = len(full_dataset)
    print(f"Full dataset size: {dataset_size}")

    # --- Dataloader for the ENTIRE dataset ---
    # We will reuse this loader for each fold's model
    full_loader = DataLoader(full_dataset,
                             batch_size=args.test_batch_size,
                             shuffle=False, # IMPORTANT: Ensure order is preserved
                             num_workers=args.num_workers,
                             pin_memory=True)

    # --- Recreate K-Fold Splits (needed ONLY for calculating TARGET_SCALE per fold) ---
    idx = np.arange(dataset_size)
    kfold = sklearn.model_selection.KFold(n_splits=args.K, shuffle=True, random_state=args.seed)
    target_scale=2099.186565

    # --- Initialize Model Structure (weights will be loaded per fold) ---
    n_classes = 1 if args.target != 'bulking' else 2
    try:
         model = timm.create_model(args.backbone, pretrained=False, num_classes=n_classes)
    except Exception as e:
        print(f"Error creating model '{args.backbone}' with {n_classes} classes: {e}")
        print("Ensure the backbone name and num_classes match the training script.")
        exit()

    model.to(device)
    model.eval() # Set model to evaluation mode

    # --- Testing Loop ---
    # Store predictions: Dictionary mapping fold number to list of predictions for all images
    all_fold_predictions = {fold: [None] * dataset_size for fold in range(args.K)}
    # Store true labels only once, as they are the same for all folds
    all_true_labels = [None] * dataset_size
    processed_true_labels = False # Flag to store true labels only on the first pass

    start_time = time.time()

    with torch.no_grad(): # Disable gradient calculations for inference
        for fold in range(args.K):
            print(f"\n--- Processing Fold {fold}: Generating predictions for ALL data ---")

            # 1. Load the specific model for this fold
            model_path = get_model_path(args, fold)
            if not os.path.exists(model_path):
                print(f"ERROR: Model file not found for fold {fold} at: {model_path}. Skipping.")
                # Keep predictions as None for this fold
                continue

            print(f"Loading model from: {model_path}")
            try:
                state_dict = torch.load(model_path, map_location=device)
                if any(key.startswith('module.') for key in state_dict.keys()):
                    print("Removing 'module.' prefix from DataParallel state_dict keys.")
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading state_dict for fold {fold}: {e}. Skipping.")
                continue

            # 2. Run inference on the ENTIRE dataset using the loaded model
            current_fold_preds = all_fold_predictions[fold] # Get the list for this fold

            for i, data in enumerate(full_loader):
                inputs, labels_true_original = data[0].to(device), data[1] # Keep true labels on CPU

                outputs = model(inputs) # Get model predictions

                # Calculate batch indices relative to the full dataset
                start_idx = i * args.test_batch_size
                end_idx = start_idx + len(inputs)
                batch_indices = list(range(start_idx, end_idx))

                # Store predictions and true labels (only once)
                preds_np = outputs.cpu().numpy()
                labels_true_np = labels_true_original.cpu().numpy()

                for j in range(len(inputs)):
                    original_idx = batch_indices[j] # Index in the full dataset
                    prediction = preds_np[j]
                    true_label = labels_true_np[j]

                    # Store true label only during the first fold's processing
                    if not processed_true_labels:
                         if args.target == 'bulking':
                              all_true_labels[original_idx] = int(true_label)
                         else:
                              all_true_labels[original_idx] = float(true_label)

                    # Process prediction
                    if args.target == 'bulking':
                         pred_value = np.argmax(prediction) # Predicted class index
                    else: # Regression
                         pred_value = prediction[0] * target_scale # Unscale using fold-specific scale

                    current_fold_preds[original_idx] = pred_value # Store prediction for this fold

            # After processing all batches for the first fold, mark true labels as done
            if fold == 0:
                processed_true_labels = True


    end_time = time.time()
    print(f"\n--- Prediction Generation Complete ---")
    print(f"Total inference time for {args.K} models across {dataset_size} images: {end_time - start_time:.2f} seconds")

    # --- Data Integrity Check ---
    num_valid_true = sum(1 for x in all_true_labels if x is not None)
    print(f"Successfully retrieved {num_valid_true} true labels out of {dataset_size}.")
    for f in range(args.K):
        num_valid_preds = sum(1 for x in all_fold_predictions[f] if x is not None)
        print(f"Fold {f}: Generated {num_valid_preds} predictions out of {dataset_size}.")

    ensemble_predictions = None
    average_pred_per_date = {}
    try:
        # 1. Simple Ensemble Average
        all_preds_array = np.array(list(all_fold_predictions.values()), dtype=float)
        ensemble_predictions = np.mean(all_preds_array, axis=0).tolist() # Get average prediction per image

        # 2. Simple Date Averaging (Only if ensemble worked and target is suitable)
        if ensemble_predictions and args.target != 'bulking':
            import os
            from collections import defaultdict

            date_predictions = defaultdict(list)
            # Assume full_dataset.samples exists and is correct
            image_paths = [item[0] for item in full_dataset.samples]

            for i, img_path in enumerate(image_paths):
                 # Assume parent directory is the date
                 date_str = os.path.basename(os.path.dirname(img_path))
                 if date_str: # Only proceed if a date string was found
                     date_predictions[date_str].append(ensemble_predictions[i])

            # Calculate average for each date
            for date_str, pred_list in date_predictions.items():
                 average_pred_per_date[date_str] = sum(pred_list) / len(pred_list)
            print(f"Finished calculating average prediction for {len(average_pred_per_date)} dates.")
    except Exception as e:
        print(f"Warning: An error occurred during simplified analysis: {e}")

    # --- Save Results ---
    # Results include true labels and a dictionary of predictions per fold
    if args.results_file:
        results_data = {
            'true_labels': all_true_labels,
            'fold_predictions': all_fold_predictions, # Dict: fold -> list_of_predictions
            'args': vars(args) # Save args used for testing for reproducibility
        }
        print(f"\nSaving detailed results (true labels and predictions per fold) to: {args.results_file}")
        os.makedirs(os.path.dirname(args.results_file) or '.', exist_ok=True)
        with open(args.results_file, 'wb') as f:
            pickle.dump(results_data, f)

    print("\n--- Analysis Notes ---")
    print("Results file contains 'true_labels' (list) and 'fold_predictions' (dictionary).")
    print("Each key in 'fold_predictions' (0 to K-1) maps to a list of predictions for the *entire* dataset made by that fold's model.")
    print("\nTesting script finished.")

