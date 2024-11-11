from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def get_symptom_model():
    from symptom_diagnosis_model import SymptomModel
    data = pd.read_csv("./dataset.csv")  # Replace with your file path

    X = data.drop(
        columns=['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8',
                 'Symptom_9'])
    y = data['Disease']

    y = y.astype('category').cat.codes
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_symptoms = StandardScaler()
    X_train = scaler_symptoms.fit_transform(X_train)

    input_size = X_train.shape[1]
    hidden_size = 64
    num_classes = len(np.unique(y))

    symptoms_model = SymptomModel(input_size, hidden_size, num_classes)
    return symptoms_model, scaler_symptoms


def get_bloodwork_model(bloodwork_input):
    from bloodwork_diagnosis_model import BloodworkModel
    data = pd.read_csv("./blood_sample_data.csv")  # Replace with your file path

    X = data.drop(columns=['White Blood Cell Count', 'Red Blood Cell Count', 'Hemoglobin', 'Platelets',
                           'Mean Corpuscular Volume (MCV)', 'Mean Corpuscular Hemoglobin (MCH)',
                           'Mean Corpuscular Hemoglobin Concentration (MCHC)', 'Red Cell Distribution Width (RDW)',
                           'Neutrophils', 'Lymphocytes', 'Monocytes', 'Eosinophils', 'Basophils'])
    y = data['Illness']

    y = y.astype('category').cat.codes
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_bloodwork = StandardScaler()
    X_train = scaler_bloodwork.fit_transform(X_train)

    input_size = X_train.shape[1]
    hidden_size = 64
    num_classes = len(np.unique(y))
    aligned_bloodwork_input = pd.get_dummies(bloodwork_input).reindex(columns=X.columns, fill_value=0)
    bloodwork_model = BloodworkModel(input_size, hidden_size, num_classes)
    return bloodwork_model, scaler_bloodwork, aligned_bloodwork_input
