import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image

# Load an example image
image_path = 'data/images_pileaute/2024-09-25/basin5/10x/25180820.JPG'  # Replace with your image path
img = Image.open(image_path)

# Define the transformationsimport cv2
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF
class ToCanny:
    def __init__(self, low_threshold=50, high_threshold=75):
        self.low = low_threshold
        self.high = high_threshold

    def __call__(self, img_tensor):
        # Convert to NumPy (C x H x W -> H x W)
        img_np = img_tensor.numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).astype(np.uint8)
        if img_np.shape[2] == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Apply Canny
        edges = cv2.Canny(img_gray, self.low, self.high)

        # Convert to tensor and add channel dim
        edges = edges.astype(np.float32) / 255.0  # Normalize to [0,1]
        edges = np.stack([edges] * 3, axis=0)

        return torch.tensor(edges, dtype=torch.float)
    
averages =  (0.485, 0.456, 0.406)
variances = (0.229, 0.224, 0.225)  
    
imgdimm = (384, 512)
train_transform = transforms.Compose([
    transforms.ToTensor(),        
    transforms.Resize(imgdimm),
    # transforms.RandomResizedCrop(imgdimm, scale=(0.8, 1.2), ratio=(1.0, 1.0)),
    # Additional transformations
    transforms.RandomApply(
        torch.nn.ModuleList([transforms.ColorJitter(
            brightness=0.2, contrast=0, saturation=0, hue=0),
            ]), p=0.5), 
    ToCanny(),
    ##### geometric transformations
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomApply(
        torch.nn.ModuleList([transforms.RandomRotation(180),
            ]), p=0.5),        
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
    #####
           
    transforms.Normalize(averages, variances),   
])

# Apply the original and transformed versions
original_image = transforms.ToTensor()(img)  # No transformation for original
transformed_image = train_transform(img)

# Plot the images side by side
plt.figure(figsize=(15, 5), dpi=200)

# Original image subplot
plt.subplot(1, 2, 1)
plt.imshow(original_image.permute(1, 2, 0))
plt.title("Original Image")
plt.axis('off')  # Turn off axis labels

# Transformed image subplot
plt.subplot(1, 2, 2)
plt.imshow(transformed_image.permute(1, 2, 0))
plt.title("Transformed Image")
plt.axis('off')  # Turn off axis labels

# Show the plot
plt.tight_layout()
plt.show()