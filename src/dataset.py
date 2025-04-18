from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from os import listdir
from PIL import Image as PImage
import numpy as np

def extract_images_and_labels(base_folder):
    """
    Extract images and labels from the specified folder.

    Parameters:
        base_folder (str): The base folder containing subfolders with images.

    Returns:
        all_images (list): A list of all extracted images.
        image_labels (numpy array): A numpy array of the corresponding labels.
    """
    # Initialize lists for images and labels
    all_images = []
    image_labels = []
    image_folders = listdir(base_folder) 

    # Save all images 
    for folder in image_folders:
        path = f"{base_folder}/{folder}"
        images_list = listdir(path)
        for image in images_list:
            img = PImage.open(f"{path}/{image}")  # open in RGB color space
            all_images.append(img)
            image_labels.append(0) #I just put all SVI labels at 0 since I don't have SVI predictions

    # Convert image labels to numpy array
    image_labels = np.array(image_labels)

    return all_images, image_labels


class MicroscopicImages(Dataset):
    def __init__(self, root, transform=None, label='SVI'):
        """
        Args:
            root (str): Path to the rooth directory containing images
            transform (callable, optional): Optional transform to be applied on a sample.
            label (str or list): Target labels for the dataset, default is 'SVI'
        """
        self.root = root
        self.transform = transform
        self.target = label
        # Use the extract_images_and_labels function to load images and labels
        self.images, self.targets = extract_images_and_labels(root) 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
          try:
            image = self.transform(image)
          except Exception as e:
            print(f"Transform failed for image at index {idx}: {e}")
            return None, None

        target = self.targets[idx].astype(np.float32)

        return image, target