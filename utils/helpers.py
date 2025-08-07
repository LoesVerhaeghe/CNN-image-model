
from os import listdir
import os
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

    return all_images, image_labels


def extract_image_paths(path_to_folders, start_folder, end_folder, magnification=10):
    """
    Extract paths from all the images from the specified folder.

    Parameters:
        base_folder (str): The base folder containing subfolders with images.
        start_folder: start date from which images need to be extracted
        end_folder: end date until which images need to be extracted
        magnification: type of magnification (10 or 40)

    Returns:
        all_images (list): A list of all extracted images.
    """
    image_folders = sorted(listdir(path_to_folders)) 

    all_paths = []

    # Select the images from start until end date
    selected_folders = [folder for folder in image_folders if start_folder <= folder <= end_folder]
    selected_folders = sorted(selected_folders)

    # Save all paths from the selected folders
    for folder in selected_folders:
        path_to_image = f"{path_to_folders}/{folder}/basin5/{magnification}x"
        images_list = sorted(listdir(path_to_image))
        for image in images_list:
            all_paths.append(f"{path_to_image}/{image}")
    return all_paths

def extract_image_paths_zurich(path_to_folders, start_folder, end_folder):
    """
    Extract paths from all the images from the specified folder.

    Parameters:
        base_folder (str): The base folder containing subfolders with images.
        start_folder: start date from which images need to be extracted
        end_folder: end date until which images need to be extracted

    Returns:
        all_images (list): A list of all extracted images.
    """
    image_folders = sorted(listdir(path_to_folders)) 

    all_paths = []

    # Select the images from start until end date
    selected_folders = [folder for folder in image_folders if start_folder <= folder <= end_folder]
    selected_folders = sorted(selected_folders)

    # Save all paths from the selected folders
    for folder in selected_folders:
        subfolders=sorted(listdir(f"{path_to_folders}/{folder}"))
        for subfolder in subfolders:
            path_to_image = f"{path_to_folders}/{folder}/{subfolder}"
            images_list = listdir(path_to_image)
            for image in images_list:
                all_paths.append(f"{path_to_image}/{image}")
    return all_paths
