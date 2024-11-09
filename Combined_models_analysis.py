import torch
import pandas as pd
from torch import nn
from bloodwork_diagnosis_model import bloodworkmodel
from symptom_diagnosis_model import model, test_loader, scaler, label_encoder
from torchvision import transforms
from PIL import Image
import numpy as np

# Ensure `new_symptoms` is a list of strings representing symptom names.
def align_symptom_input(user_symptoms, symptom_index_map):
    # Create a zero array with the same length as the number of features in the training data
    aligned_input = np.zeros(len(symptom_index_map))

    # Set the corresponding index to 1 for each symptom in the user's input
    for symptom in user_symptoms:
        if symptom in symptom_index_map:
            index = symptom_index_map[symptom]
            aligned_input[index] = 1
        else:
            print(f"Warning: Symptom '{symptom}' not found in training data features.")

    return aligned_input

# Load models as before
def load_models():
    symptoms_model = torch.load('./symptom_diagnosis_model.pth')
    bloodwork_model = torch.load('./bloodwork_diagnosis_model.pth')
    scans_model = torch.load('./scan_classification_model.pth')
    print("Models loaded.")
    return symptoms_model, bloodwork_model, scans_model

# Function to evaluate a model's accuracy
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in data_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Single-image evaluation function for scan model
def evaluate_single_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image = image.to(device)

    # Predict labels for the image
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.sigmoid(outputs)  # Convert logits to probabilities

    predicted_labels = (probabilities > 0.5).squeeze().cpu().numpy()
    return predicted_labels

# Main function to evaluate all inputs
def eval_with_inputs(name, new_symptoms, bloodwork, scan):
    symptoms_model, bloodwork_model, scans_model = load_models()

    # Prepare the symptoms dataset
    symptoms_data = pd.read_csv("./dataset.csv")
    X = symptoms_data.drop(
        columns=['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8',
                 'Symptom_9'])
    X = pd.get_dummies(X)  # One-hot encoding

    # Create a mapping of symptom names to their indices in the one-hot encoded DataFrame
    symptom_index_map = {symptom: idx for idx, symptom in enumerate(X.columns)}

    # Check and align the symptom input
    print("Symptom indices mapping:", symptom_index_map)
    aligned_input = align_symptom_input(new_symptoms, symptom_index_map)
    print("Aligned symptom input:", aligned_input)

    # Scale and convert to tensor
    aligned_input_scaled = scaler.transform([aligned_input])  # Transform expects a 2D array
    new_symptoms_tensor = torch.tensor(aligned_input_scaled, dtype=torch.float32)

    # Prepare bloodwork tensor
    new_bloodwork_tensor = torch.tensor(bloodwork, dtype=torch.float32)

    # Begin testing with sample data
    symptoms_model.eval()
    with torch.no_grad():
        output = symptoms_model(new_symptoms_tensor)
        _, predicted_class = torch.max(output, 1)
        predicted_classes = predicted_class.tolist()
        predicted_diseases_by_symptoms = label_encoder.inverse_transform(predicted_classes)

    for disease in predicted_diseases_by_symptoms:
        print(f"Name: {name}")
        print(f"Predicted Disease: {disease}")

    # Evaluate bloodwork model
    bloodwork_model.eval()
    with torch.no_grad():
        output = bloodwork_model(new_bloodwork_tensor)
        _, predicted_class = torch.max(output, 1)
        predicted_classes = predicted_class.tolist()
        predicted_illness_by_bloodwork = label_encoder.inverse_transform(predicted_classes)

    for illness in predicted_illness_by_bloodwork:
        print(f"Name: {name}")
        print(f"Predicted Illness by Bloodwork: {illness}")

    # Evaluate scan model
    scan_prediction = evaluate_single_image(scans_model, scan)

    print("Predictions from symptoms:", predicted_diseases_by_symptoms)
    print("Predictions from bloodwork:", predicted_illness_by_bloodwork)
    print("Predictions for the test image:", scan_prediction)
