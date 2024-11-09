#here we can take in the file and then add tags to the files that align with the symptoms
# for example a scan could be tagged with 'migrane, cough, sore throat'\
# we need more investigation on this topic before deciding
import random

import kagglehub
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps
import pandas as pd
from torchvision.models import ResNet50_Weights

# Define the custom Dataset class outside of main()
class MedicalImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

def main():
    os.environ['KAGGLE_CONFIG_DIR'] = 'C:/Users/peach/kaggle/kaggle.json'

    #replace with permanent server path later
    path = 'C:\\Users\\peach\\.cache\\kagglehub\\datasets\\nih-chest-xrays\\data\\versions\\3'

    # this only downloads if you have a kaggle API key in your system check kaggle docs for a guide
    if path == False or path == None:
        path = kagglehub.dataset_download("nih-chest-xrays/data")
    print("Path to dataset files:", path)

    # Step 1: Load and process CSV data
    csv_data = pd.read_csv(f'{path}\\Data_Entry_2017.csv')

    # Step 2: Define unique labels and create an encoding dictionary
    all_labels = sorted(set(label for labels in csv_data['Finding Labels'] for label in labels.split('|')))
    label_to_index = {label: idx for idx, label in enumerate(all_labels)}
    num_classes = len(all_labels)


    # Step 3: Encode labels as binary vectors
    def encode_labels(label_str):
        labels = label_str.split('|')
        label_vector = np.zeros(num_classes, dtype=np.float32)
        for label in labels:
            if label in label_to_index:
                label_vector[label_to_index[label]] = 1
        return label_vector


    # Add encoded labels as a new column in csv_data
    csv_data['Encoded Labels'] = csv_data['Finding Labels'].apply(encode_labels)

    # Step 4: Create a dictionary mapping image filename to encoded labels
    label_dict = {row['Image Index']: row['Encoded Labels'] for _, row in csv_data.iterrows()}

    # Step 5: Collect image paths and match with labels for training

    # Function to select a subset of image files from each folder
    def get_subset_of_images(files, max_images_per_folder=100):
        if len(files) > max_images_per_folder:
            return random.sample(files, max_images_per_folder)  # Randomly select
        return files

    # Limit the number of images per folder (e.g., 100 images per folder)
    max_images_per_folder = 100

    train_image_paths = []
    train_labels = []

    with open(f'{path}\\test_list.txt', 'r') as file:
        train_files = [line.strip() for line in file]

    for filename in get_subset_of_images(train_files, max_images_per_folder):
        for i in range(1, 14):  # Loop through all 13 folders
            folder_path = os.path.join(path, f'images_00{i}', 'images')
            image_path = os.path.join(folder_path, filename)

            if os.path.isfile(image_path):
                train_image_paths.append(image_path)
                train_labels.append(label_dict[filename])
                break
    # Step 6: Collect image paths and match with labels for validation
    val_image_paths = []
    val_labels = []

    with open(f'{path}\\train_val_list.txt', 'r') as file:
        val_files = [line.strip() for line in file]

    for filename in get_subset_of_images(val_files, max_images_per_folder):
        for i in range(1, 14):
            folder_path = os.path.join(path, f'images_00{i}', 'images')
            image_path = os.path.join(folder_path, filename)

            if os.path.isfile(image_path):
                val_image_paths.append(image_path)
                val_labels.append(label_dict[filename])
                break

    # Step 7: Convert label lists to numpy arrays for consistency
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)


    # print("training: \n", train_image_paths, "validation: \n", val_image_paths)

    # Load a pretrained ResNet and modify the final layer for binary/multiclass classification
    # Load the ResNet50 model
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Check if a GPU is available
    if torch.cuda.is_available():
        # Move the model to the GPU
        device = torch.device("cuda")
        model = model.to(device)
        print("Model loaded on GPU")
    else:
        print("No GPU available, using CPU")
    num_ftrs = model.fc.in_features
    criterion = nn.BCEWithLogitsLoss()
    model.fc = nn.Linear(num_ftrs, num_classes)  # Match the number of classes

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    # Define transformations for training and validation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # Step 8: Instantiate the datasets with matching paths and labels
    train_dataset = MedicalImageDataset(train_image_paths, train_labels, transform=transform)
    val_dataset = MedicalImageDataset(val_image_paths, val_labels, transform=transform)

    print("train dataset: \n",train_dataset, "val dataset: \n",val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Hyperparameters
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 32

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Suitable for multi-label classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Move model to the selected device (GPU if available)
    model = model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for images, labels in train_loader:
            # Move images and labels to device
            images = images.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)  # Accumulate batch loss

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Evaluation on validation set
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            val_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                # Convert model outputs to probabilities and then predictions (0 or 1)
                predictions = torch.sigmoid(outputs) > 0.5

                # Update counts for accuracy
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.numel()

            # Average validation loss
            avg_val_loss = val_loss / len(val_loader.dataset)
            accuracy = correct_predictions / total_samples

            print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

        model_save_path = 'scan_classification_model.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
