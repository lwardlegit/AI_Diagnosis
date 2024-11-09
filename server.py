import json

from flask import Flask, request, jsonify
from flask_cors import CORS
from Combined_models_analysis import eval_with_inputs
import csv

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes


@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    unique_symptoms = set()  # Use a set to automatically handle duplicates

    with open('./dataset.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            # Extract symptoms starting from column 2 (index 1) onward
            symptoms = row[1:]
            # Filter out empty values and add symptoms to the set
            unique_symptoms.update([symptom.strip() for symptom in symptoms if symptom.strip()])
    print(unique_symptoms)
    unique_symptoms = jsonify(list(unique_symptoms))
    print(unique_symptoms)
    return unique_symptoms


@app.route('/submit', methods=['POST'])
def submit_form():
    name = request.form.get('name')
    symptoms = request.form.get('symptoms')
    scan = request.files.get('file')
    bloodwork = request.files.get('bloodwork')

    print("Name:", name)
    print("Symptoms:", symptoms)
    print("Scan:", scan.filename if scan else "No file")
    print("Bloodwork:", bloodwork.filename if scan else "No file")
    result = eval_with_inputs(name, symptoms, scan, bloodwork)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
