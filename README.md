# CNN Deep Learning for Wastewater Image Analysis

Adapted from [wastewater_vision](https://github.com/scabini/wastewater_vision), this repository provides CNN-based deep learning models for wastewater image classification and prediction, with and without transfer learning.  
Several changes have been made to improve generalization and reduce overfitting.

---

## Key Modifications
- **Data augmentation**: added Gaussian blur and other transformations  
- **Model head**: two fully connected layers with dropout (instead of one)  
- **Weight decay**: `1e-6` (instead of `1e-8` in transfer learning mode)  
- **Learning rate scheduling**: `ReduceLROnPlateau` on validation loss (instead of cosine annealing)  
- **Cross-validation**: no shuffling in k-fold splits  

---

## Archived Experiments
Older experiments are stored in the `archive/` folder:
- **Edge detection preprocessing** – apply edge detection before CNN training  
- **Extra input features** – combine image features with additional inputs in the head  
- **Layer freezing** – freeze CNN layers for 5 epochs, train head, then unfreeze  
- **No feature extraction** – CNN without pre-extraction; overfitting-mitigation changes not applied  

---

## Usage
The CNN can be used in two ways:
1. **Direct prediction** – predict the target variable directly from images.  
2. **Feature extraction for temporal models** – extract features before the head and pass them to an attention-based pooling + LSTM model to analyze daily images with a temporal component.  

