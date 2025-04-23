
from pathlib import Path
import pandas as pd
from os import listdir
from PIL import Image as PImage

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


def extract_image_paths(path_to_folders, image_type='all', magnification=10):
    """
    Extract paths from all the images from the specified folder.

    Parameters:
        base_folder (str): The base folder containing subfolders with images.
        image_type (str): The type of images to extract: 'all', 'old', or 'new', train or test. Default is 'all'.
        old refers to old microscope in the lab, new refers to new microscope
        magnification: type of magnification (10 or 40)

    Returns:
        all_images (list): A list of all extracted images.
    """
    target_folder_microscope_type='2024-01-26'
    target_folder_dataset_type='2024-09-24'
    image_folders = sorted(listdir(path_to_folders)) 

    all_paths = []

    # Define the condition for folder selection based on image_type
    if image_type == 'all':
        selected_folders = image_folders
    elif image_type == 'old':
        selected_folders = [folder for folder in image_folders if folder < target_folder_microscope_type]
    elif image_type == 'new':
        selected_folders = [folder for folder in image_folders if folder >= target_folder_microscope_type]
    elif image_type == 'train':
        selected_folders = [folder for folder in image_folders if folder <= target_folder_dataset_type]
    elif image_type == 'test':
        selected_folders = [folder for folder in image_folders if folder > target_folder_dataset_type]
    else:
        raise ValueError("Invalid image_type. Choose from 'all', 'old', or 'new' or 'train' or 'test'.")

    selected_folders = sorted(selected_folders)

    # Save all paths from the selected folders
    for folder in selected_folders:
        path_to_image = f"{path_to_folders}/{folder}/basin5/{magnification}x"
        images_list = sorted(listdir(path_to_image))
        for image in images_list:
            all_paths.append(f"{path_to_image}/{image}")
    return all_paths
