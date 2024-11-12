import torch
import pandas as pd
import re
from torch import nn
from torchvision.models import ResNet50_Weights

from bloodwork_diagnosis_model import bloodworkmodel, BloodworkModel, predict_illness_bloodwork
from model_helper_functions import get_bloodwork_model, get_symptom_model
from symptom_diagnosis_model import model, test_loader, scaler, label_encoder, SymptomModel, symptom_index_map
from torchvision import transforms, models
from PIL import Image
import numpy as np

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


def read_csv_with_multiple_encodings(file_path):
    encodings = ["utf-8", "ISO-8859-1", "latin1", "utf-16", "utf-8-sig"]
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully read the file with encoding: {encoding}")
            return df  # Return the DataFrame if successful
        except UnicodeDecodeError:
            print(f"Failed to read with encoding: {encoding}")

    raise ValueError("None of the encodings were successful in reading the file.")

# Single-image evaluation function for scan model
def evaluate_single_image(image_path):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.load_state_dict(torch.load("scan_classification_model.pth"))
    model.eval()
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


def predict_disease(symptom_strings, symptom_index_map, model, scaler, label_encoder):
    # Create an aligned input vector for the symptoms
    input_vector = np.zeros(len(symptom_index_map))
    for symptom in symptom_strings:
        if symptom in symptom_index_map:
            index = symptom_index_map[symptom]
            input_vector[index] = 1  # Mark the presence of the symptom

    # Scale the input
    input_vector = scaler.transform([input_vector])  # Scaling expects a 2D array

    # Convert to tensor for model input
    input_tensor = torch.tensor(input_vector, dtype=torch.float32)

    # Make the prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)

    # Convert numeric prediction back to disease name
    predicted_disease = label_encoder.inverse_transform(predicted_class.numpy())

    return predicted_disease[0]

# Main function to evaluate all inputs
def eval_with_inputs(name, new_symptoms, bloodwork, scan):
    #symptoms we can have - new symptoms length so our input size is correct
    row = 17 - len(new_symptoms)
    while len(new_symptoms) < row:
        new_symptoms.append(None)
    print(new_symptoms)


    # Evaluate symptom model
    predicted_disease = predict_disease(new_symptoms, symptom_index_map, model, scaler, label_encoder)
    print("Name: ", name,"\n")
    print("Predicted Disease:", predicted_disease)

    #Evaluate bloodwork model
    predicted_illness = predict_illness_bloodwork(bloodwork)
    print("Predicted Disease:", predicted_illness)

    # Evaluate scan model
    scan_prediction = evaluate_single_image(scan)
    print("Predictions for the test image:", scan_prediction)



