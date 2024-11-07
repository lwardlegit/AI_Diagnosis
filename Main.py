import torch
import pandas as pd
from torch import nn
from bloodwork_diagnosis_model import bloodworkmodel
from symptom_diagnosis_model import model, test_loader, scaler, label_encoder

# Load Symptoms dataset
data = pd.read_csv("./dataset.csv")
y = data['Disease']

# Load bloodwork dataset
data = pd.read_csv("./blood_sample_data.csv")
y2 = data['Illness']

# Load scans dataset

def load_model():
    model = torch.load('./symptom_diagnosis_model.pth')
    print("Model loaded.")
    return model

# this is a test to see how accurate the model is
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

# Evaluate on test set
test_accuracy = evaluate_model(model, test_loader)
print(f'Test Accuracy: {test_accuracy:.2f}%')


# test the model against user input

def eval_with_inputs(name,new_symptoms,scan,bloodwork):
    new_symptoms = scaler.transform(new_symptoms)
    new_symptoms_tensor = torch.tensor(new_symptoms, dtype=torch.float32)

    new_bloodwork_tensor = torch.tensor(bloodwork, dtype=torch.float32)

    print("BEGINNING TEST WITH SAMPLE DATA...")
    model.eval()
    with torch.no_grad():
        output = model(new_symptoms_tensor)
        _, predicted_class = torch.max(output, 1)
        predicted_classes = predicted_class.tolist()  # Convert tensor to a list
        predicted_diseases = label_encoder.inverse_transform(predicted_classes)  # Inverse transform

    # Print predicted diseases
    # in the data CSV there are multiple entries for some illnesses Ex: any answer returned as 0-9 would represent "Fungal infection"
    for disease in predicted_diseases:
        print(f"name: {name}")
        print(f"Predicted Disease: {disease}")
        print(f"tensor item: {y[disease]}")


    bloodworkmodel.eval()
    with torch.no_grad():
        output = bloodworkmodel(new_bloodwork_tensor)
        _, predicted_class = torch.max(output, 1)
        predicted_classes = predicted_class.tolist()  # Convert tensor to a list
        predicted_illness = label_encoder.inverse_transform(predicted_classes)  # Inverse transform


    for illness in predicted_illness:
        print(f"name: {name}")
        print(f"Predicted Illness by bloodwork: {illness}")
        print(f"tensor item: {y2[illness]}")


    #scan model will evaluate here

    #if we have all 3 answers we need to create a new CSV with correct answers from all 3 categories