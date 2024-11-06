import json

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes


@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    symptoms = ["Cough", "Fever", "Headache", "Sore Throat"]
    return jsonify(symptoms)


@app.route('/submit', methods=['POST'])
def submit_form():
    name = request.form.get('name')
    symptoms = request.form.get('symptoms')
    file = request.files.get('file')

    symptoms = json.loads(symptoms)  # If needed

    print("Name:", name)
    print("Symptoms:", symptoms)
    print("File:", file.filename if file else "No file")
    return jsonify({"status": "success", "message": "Form received!"})


if __name__ == '__main__':
    app.run(debug=True)
