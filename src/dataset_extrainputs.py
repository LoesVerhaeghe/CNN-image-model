from utils.helpers import extract_image_paths, extract_image_paths_zurich, extract_image_paths_pantarein, extract_image_paths_bath
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image as PImage
import os

class MicroscopicImages(Dataset):
    def __init__(self, root, start_folder, end_folder, magnification, label_path, extrainput_path, transform=None):
        """
        Args:
            root (str): Path to the root directory containing images.
            magnification (str or int): Magnification filter used in image path extraction.
            label_path (str): Path to CSV file containing image labels with dates as index.
            image_type (str): type of dataset that needs to be extracted: 'all', 'old', or 'new', train or test
            extrainput_path (str): Path to CSV file containing extra input data that can be added in the last layers of the NN (e.g. COD)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.targets = []
        # self.extrainputs = [] 

        # Load CSV file containing image paths (or filenames) and labels
        self.labels_df = pd.read_csv(label_path, index_col=0)
        self.extrainputs_df = pd.read_csv(extrainput_path, index_col=0)
        
        self.samples=[]
        self.image_paths = extract_image_paths(root, start_folder=start_folder, end_folder=end_folder, magnification=magnification)
        for image_path in self.image_paths:
            # Extract date from path
            parts = image_path.split(os.sep)
            date = next((p for p in parts if p[:4].isdigit() and p[4] == '-'), None)
            
            if date is None:
                print(f"Could not extract date from path: {image_path}, skipping")
                continue

            if date not in self.labels_df.index or date not in self.extrainputs_df.index:
                print(f"Date {date} not in label or extrainput CSV, skipping")
                continue

            label = self.labels_df.loc[date].values[0]  
            extrainput= self.extrainputs_df.loc[date].values
            self.targets.append(label)
            self.samples.append((image_path, label, extrainput))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label, extrainput = self.samples[idx]
        
        try:
            image = PImage.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load image at index {idx} (path: {image_path}): {e}")
            return None

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Transform failed for image at index {idx}: {e}")
                return None

        return image, label, extrainput
    
###############################################################################################################################

class MicroscopicImagesZurich(Dataset):
    def __init__(self, root, start_folder, end_folder, label_path, extrainput_path, transform=None):
        """
        Args:
            root (str): Path to the root directory containing images.
            magnification (str or int): Magnification filter used in image path extraction.
            label_path (str): Path to CSV file containing image labels with dates as index.
            image_type (str): type of dataset that needs to be extracted: 'all', 'old', or 'new', train or test
            extrainput_path (str): Path to CSV file containing extra input data that can be added in the last layers of the NN (e.g. COD)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transform = transform
        self.label_path=label_path
        self.targets = []

        # Load CSV file containing image paths (or filenames) and labels
        self.labels_df = pd.read_csv(label_path, index_col=0)
        self.extrainputs_df = pd.read_csv(extrainput_path, index_col=0)
        
        self.samples=[]
        self.image_paths = extract_image_paths_zurich(root, start_folder=start_folder, end_folder=end_folder)

        for image_path in self.image_paths:
            # Extract date from path
            parts = image_path.split(os.sep)
            date = next((p for p in parts if p[:4].isdigit() and p[4] == '-'), None)
            
            if date is None:
                print(f"Could not extract date from path: {image_path}, skipping")
                self.targets.append(None)
                continue

            if date not in self.labels_df.index or date not in self.extrainputs_df.index:
                print(f"Date {date} not in label or extrainput CSV, skipping")
                self.targets.append(None)
                continue

            label = self.labels_df.loc[date].values[0]  
            extrainput= self.extrainputs_df.loc[date].values
            self.targets.append(label)
            self.samples.append((image_path, label, extrainput))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label, extrainput = self.samples[idx]
        
        try:
            image = PImage.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load image at index {idx} (path: {image_path}): {e}")
            return None

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Transform failed for image at index {idx}: {e}")
                return None

        return image, label, extrainput

class MicroscopicImagesPantarein(Dataset):
    def __init__(self, root, start_folder, end_folder, label_path, extrainput_path, transform=None):
        """
        Args:
            root (str): Path to the root directory containing images.
            magnification (str or int): Magnification filter used in image path extraction.
            label_path (str): Path to CSV file containing image labels with dates as index.
            image_type (str): type of dataset that needs to be extracted: 'all', 'old', or 'new', train or test
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transform = transform
        self.label_path = label_path
        self.extrainput_path = extrainput_path
        self.image_paths = extract_image_paths_pantarein(root, start_folder=start_folder, end_folder=end_folder)
        self.targets = []
        # Load CSV file containing image paths (or filenames) and labels
        self.labels_df = pd.read_csv(label_path, index_col=0)
        self.extrainput_df= pd.read_csv(extrainput_path, index_col=0)
         
        for image_path in self.image_paths:
            # Extract date
            parts = image_path.split(os.sep)
            date = next((p for p in parts if p[:4].isdigit() and p[4] == '-'), None)
            
            if date is None:
                print(f"Could not extract date from path: {image_path}")
                self.targets.append(None)
                continue

            if date not in self.labels_df.index:
                print(f"Date {date} not in label CSV")
                self.targets.append(None)
                continue

            label = self.labels_df.loc[date].values[0] 
            extrainput = self.extrainput_df.loc[date].values  
            self.targets.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            image = PImage.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load image at index {idx} (path: {image_path}): {e}")
            return None

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Transform failed for image at index {idx}: {e}")
                return None

        # Get the date from the folder structure: 
        parts = image_path.split(os.sep)
        date = None
        for part in parts:
            if part[:4].isdigit() and part[4] == '-':  # e.g. '2023-11-02'
                date = part
                break

        if date is None:
            print(f"Could not find date folder in path: {image_path}")
            return None

        if date not in self.labels_df.index:
            print(f"Date {date} not found in label CSV")
            return None

        label = self.labels_df.loc[date].values[0]  # Adjust if multiple columns
        extrainput = self.extrainput_df.loc[date].values


        return image, label, extrainput, date
    



class MicroscopicImagesBath(Dataset):
    def __init__(self, root, start_folder, end_folder, label_path, extrainput_path, transform=None):
        """
        Args:
            root (str): Path to the root directory containing images.
            magnification (str or int): Magnification filter used in image path extraction.
            label_path (str): Path to CSV file containing image labels with dates as index.
            image_type (str): type of dataset that needs to be extracted: 'all', 'old', or 'new', train or test
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transform = transform
        self.label_path = label_path
        self.extrainput_path = extrainput_path
        self.image_paths = extract_image_paths_bath(root, start_folder=start_folder, end_folder=end_folder)
        self.targets = []
        # Load CSV file containing image paths (or filenames) and labels
        self.labels_df = pd.read_csv(label_path, index_col=0)
        self.extrainput_df= pd.read_csv(extrainput_path, index_col=0)
         
        for image_path in self.image_paths:
            # Extract date
            parts = image_path.split(os.sep)
            date = next((p for p in parts if p[:4].isdigit() and p[4] == '-'), None)
            
            if date is None:
                print(f"Could not extract date from path: {image_path}")
                self.targets.append(None)
                continue

            if date not in self.labels_df.index:
                print(f"Date {date} not in label CSV")
                self.targets.append(None)
                continue

            label = self.labels_df.loc[date].values[0] 
            extrainput = self.extrainput_df.loc[date].values  
            self.targets.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            image = PImage.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load image at index {idx} (path: {image_path}): {e}")
            return None

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Transform failed for image at index {idx}: {e}")
                return None

        # Get the date from the folder structure: 
        parts = image_path.split(os.sep)
        date = None
        for part in parts:
            if part[:4].isdigit() and part[4] == '-':  # e.g. '2023-11-02'
                date = part
                break

        if date is None:
            print(f"Could not find date folder in path: {image_path}")
            return None

        if date not in self.labels_df.index:
            print(f"Date {date} not found in label CSV")
            return None

        label = self.labels_df.loc[date].values[0]  # Adjust if multiple columns
        extrainput = self.extrainput_df.loc[date].values


        return image, label, extrainput, date
    


class MicroscopicImagesBathCompression(Dataset):
    def __init__(self, root, start_folder, end_folder, label_path, extrainput_path, transform=None):
        """
        Args:
            root (str): Path to the root directory containing images.
            magnification (str or int): Magnification filter used in image path extraction.
            label_path (str): Path to CSV file containing image labels with dates as index.
            image_type (str): type of dataset that needs to be extracted: 'all', 'old', or 'new', train or test
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transform = transform
        self.label_path = label_path
        self.extrainput_path = extrainput_path
        self.image_paths = extract_image_paths_bath(root, start_folder=start_folder, end_folder=end_folder)
        self.targets = []
        # Load CSV file containing image paths (or filenames) and labels
        self.labels_df = pd.read_csv(label_path, index_col=0)
        self.extrainput_df= pd.read_csv(extrainput_path, index_col=0)
          
        for image_path in self.image_paths:
            # Extract date
            parts = image_path.split(os.sep)
            date = next((p for p in parts if p[:4].isdigit() and p[4] == '-'), None)
            
            if date is None:
                print(f"Could not extract date from path: {image_path}")
                self.targets.append(None)
                continue

            if date not in self.labels_df.index:
                print(f"Date {date} not in label CSV")
                self.targets.append(None)
                continue

            label = self.labels_df.loc[date].values[0:3]  # adjust column if needed
            extrainput = self.extrainput_df.loc[date].values 
            self.targets.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            image = PImage.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load image at index {idx} (path: {image_path}): {e}")
            return None

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Transform failed for image at index {idx}: {e}")
                return None

        # Get the date from the folder structure: 
        parts = image_path.split(os.sep)
        date = None
        for part in parts:
            if part[:4].isdigit() and part[4] == '-':  # e.g. '2023-11-02'
                date = part
                break

        if date is None:
            print(f"Could not find date folder in path: {image_path}")
            return None

        if date not in self.labels_df.index:
            print(f"Date {date} not found in label CSV")
            return None

        label = self.labels_df.loc[date].values[0:3]#.astype(np.float32)  # Adjust if multiple columns
        extrainput = self.extrainput_df.loc[date].values

        return image, label, extrainput, date