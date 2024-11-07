#here we can take in the file and then add tags to the files that align with the symptoms
# for example a scan could be tagged with 'migrane, cough, sore throat'\
# we need more investigation on this topic before deciding

import kagglehub
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd

os.environ['KAGGLE_CONFIG_DIR'] = 'C:/Users/peach/kaggle/kaggle.json'

#replace with permanent server path later
path = 'C:\\Users\\peach\\.cache\\kagglehub\\datasets\\nih-chest-xrays\\data\\versions\\3'

# this only downloads if you have a kaggle API key in your system check kaggle docs for a guide
if path == False or path == None:
    path = kagglehub.dataset_download("nih-chest-xrays/data")
print("Path to dataset files:", path)

#Labels
csv_data = pd.read_csv(f'{path}\\Data_Entry_2017.csv')
labels = csv_data['Finding Labels'].unique()
print(labels)

# Test Images
with open(f'{path}\\test_list.txt', 'r') as file:
    lines = file.readlines()
lines = [line.strip() for line in lines]

# Initialize a list to store the paths or content of the images
images_for_training = []

# Iterate over folders to locate and collect the images
for line in lines:
    # Check across multiple folders (images_001, images_002, etc.)
    for i in range(1, 11):  # Adjust range as needed based on your folder structure
        folder_path = os.path.join(path, f'images_00{i}', 'images')
        image_path = os.path.join(folder_path, f'{line}')

        # Check if the file exists before attempting to open
        if os.path.isfile(image_path):
            images_for_training.append(image_path)  # Store the path if found
            break  # Stop searching other folders once the image is found

print(images_for_training)

#Validation Images

with open(f'{path}\\train_val_list.txt', 'r') as file:
    lines = file.readlines()
lines = [line.strip() for line in lines]

# Initialize a list to store the paths or content of the images
images_for_train_validation  = []

# Iterate over folders to locate and collect the images
for line in lines:
    # Check across multiple folders (images_001, images_002, etc.)
    for i in range(1, 11):  # Adjust range as needed based on your folder structure
        folder_path = os.path.join(path, f'images_00{i}', 'images')
        image_path = os.path.join(folder_path, f'{line}')

        # Check if the file exists before attempting to open
        if os.path.isfile(image_path):
            images_for_train_validation.append(image_path)  # Store the path if found
            break  # Stop searching other folders once the image is found

print(images_for_train_validation)



# Load a pretrained ResNet and modify the final layer for binary/multiclass classification
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Example: 2 classes (e.g., disease present or not)
train_labels = ['']

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Define transformations for training and validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom Dataset class (assuming X-ray/CT images and labels are available)
class MedicalImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = path
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# Instantiate dataset and dataloaders
train_dataset = MedicalImageDataset(images_for_training, train_labels, transform=transform)
val_dataset = MedicalImageDataset(images_for_train_validation, train_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

