import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# Load your dataset
data = pd.read_csv("./blood_sample_data.csv")  # Replace with your file path

# Define feature and target columns
X = data.drop(columns=['White Blood Cell Count','Red Blood Cell Count','Hemoglobin','Platelets','Mean Corpuscular Volume (MCV)',
                       'Mean Corpuscular Hemoglobin (MCH)','Mean Corpuscular Hemoglobin Concentration (MCHC)',
                       'Red Cell Distribution Width (RDW)','Neutrophils','Lymphocytes','Monocytes','Eosinophils','Basophils'])
y = data['Illness']  # Target labels

# Encode target labels to numerical values if they are categorical
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X = pd.get_dummies(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define custom Dataset
class BloodworkDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Instantiate the dataset
train_dataset = BloodworkDataset(X_train, y_train)
test_dataset = BloodworkDataset(X_test, y_test)

# DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model
class BloodworkModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BloodworkModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Hyperparameters
input_size = X_train.shape[1]
hidden_size = 64
num_classes = len(np.unique(y_encoded))
bloodworkmodel = BloodworkModel(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(bloodworkmodel.parameters(), lr=0.001)

# Training the model
num_epochs = 20
for epoch in range(num_epochs):
    bloodworkmodel.train()
    total_loss = 0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = bloodworkmodel(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(bloodworkmodel.state_dict(), "bloodwork_diagnosis_model.pth")
print("Model saved as 'bloodwork_diagnosis_model.pth'")

def predict_illness_bloodwork(bloodwork):
    # Load the saved model and set to evaluation mode
    bloodworkmodel.load_state_dict(torch.load("bloodwork_diagnosis_model.pth"))
    bloodworkmodel.eval()

    # Load the input sample
    sample_data = pd.read_csv(bloodwork)

    # Align the input sample columns with the training features
    expected_columns = X.columns  # Columns used during model training
    aligned_sample = pd.DataFrame(0, index=[0], columns=expected_columns)

    # Insert bloodwork values into the aligned sample
    for col in sample_data.columns:
        if col in aligned_sample.columns:
            aligned_sample[col] = sample_data[col].values

    # Standardize the input sample using the scaler from training
    aligned_sample = scaler.transform(aligned_sample)

    # Convert the input sample to a PyTorch tensor
    sample_tensor = torch.tensor(aligned_sample, dtype=torch.float32)

    # Evaluate the model on the input sample
    with torch.no_grad():
        outputs = bloodworkmodel(sample_tensor)
        _, predicted_class = torch.max(outputs, 1)

    # Decode the predicted class index to the original illness label
    predicted_label = label_encoder.inverse_transform(predicted_class.numpy())  # Convert tensor to numpy array

    # Display the predicted illness
    print(f"Predicted diagnosis for the input sample: {predicted_label[0]}")
