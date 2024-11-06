import React, { useState, useEffect } from 'react';
import axios from 'axios';

const SymptomsForm = () => {
  const [symptoms, setSymptoms] = useState([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [name, setName] = useState('');
  const [file, setFile] = useState(null);

  // Fetch symptoms from the server
  useEffect(() => {
    const fetchSymptoms = async () => {
      try {
        const response = await axios.get('http://localhost:5000/symptoms');
        setSymptoms(response.data);
      } catch (error) {
        console.error('Error fetching symptoms:', error);
      }
    };
    fetchSymptoms();
  }, []);

  // Handle symptom selection
  const handleSymptomClick = (symptom) => {
    setSelectedSymptoms((prevSelected) => {
      if (prevSelected.includes(symptom)) {
        return prevSelected.filter((item) => item !== symptom);
      } else {
        return [...prevSelected, symptom];
      }
    });
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('name', name);
    formData.append('file', file);
    formData.append('symptoms', JSON.stringify(selectedSymptoms));

    try {
      await axios.post('http://localhost:5000/submit', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      alert('Form submitted successfully!');
    } catch (error) {
      console.error('Error submitting form:', error);
    }
  };

  return (
    <div>
      <h1>Select Symptoms</h1>
      <div>
        {symptoms.map((symptom) => (
          <button
            key={symptom}
            onClick={() => handleSymptomClick(symptom)}
            style={{
              backgroundColor: selectedSymptoms.includes(symptom) ? 'lightblue' : 'lightgray',
              margin: '5px',
            }}
          >
            {symptom}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit}>
        <div>
          <label>Name:</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
          />
        </div>

        <div>
          <label>Upload File:</label>
          <input
            type="file"
            onChange={(e) => setFile(e.target.files[0])}
            required
          />
        </div>

        <button type="submit">Submit</button>
      </form>
    </div>
  );
};

export default SymptomsForm;
