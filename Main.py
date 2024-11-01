# model.py
import torch
from torch import nn

from symptom_diagnosis_model import model, test_loader, scaler, label_encoder

# Define a global variable to store form data from user input
global_form_data = {}

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

# we can save it if we need to improve accuracy per run
# torch.save(model.state_dict(), "symptom_diagnosis_model.pth")
# print("Model saved.")


# test the model against new data

#### Example input: Replace with actual symptom values as a list
# we probably need to let the user select symptoms from a list since if a user types them in
# it won't encode to the same value for now we can just make a dummy

new_symptoms = [[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]]  # Example data; replace with actual symptoms

# Preprocess and convert to tensor
new_symptoms = scaler.transform(new_symptoms)
new_symptoms_tensor = torch.tensor(new_symptoms, dtype=torch.float32)

print("BEGINNING TEST WITH SAMPLE DATA...")
model.eval()
with torch.no_grad():
    output = model(new_symptoms_tensor)
    _, predicted_class = torch.max(output, 1)
    predicted_disease = label_encoder.inverse_transform([predicted_class.item()])

print(f"Predicted Disease: {predicted_disease[0]}")
