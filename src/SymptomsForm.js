import React, { useState, useEffect } from 'react';
import axios from 'axios';

const SymptomsForm = () => {
  const [symptoms, setSymptoms] = useState([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [name, setName] = useState('');
  const [scanFile, setScanFile] = useState(null);
  const [bloodworkFile, setBloodworkFile] = useState(null);

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

  const handleSymptomClick = (symptom) => {
    setSelectedSymptoms((prevSelected) => {
      if (prevSelected.includes(symptom)) {
        return prevSelected.filter((item) => item !== symptom);
      } else {
        return [...prevSelected, symptom];
      }
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('name', name);
    formData.append('file', scanFile);
    formData.append('bloodwork', bloodworkFile);
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
    <div style={styles.container}>
      <h1 style={styles.heading}>Select Symptoms</h1>

      <div style={styles.symptomsContainer}>
        {symptoms.map((symptom) => (
          <button
            key={symptom}
            onClick={() => handleSymptomClick(symptom)}
            style={{
              ...styles.symptomButton,
              backgroundColor: selectedSymptoms.includes(symptom) ? '#6baed6' : '#e6f2ff',
            }}
          >
            {symptom}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} style={styles.form}>
        <div style={styles.inputGroup}>
          <label style={styles.label}>Name:</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            style={styles.input}
          />
        </div>

        <div style={styles.inputGroup}>
          <label style={styles.label}>Upload Scan:</label>
          <input
            type="file"
            onChange={(e) => setScanFile(e.target.files[0])}
            required
            style={styles.input}
          />
        </div>

        <div style={styles.inputGroup}>
          <label style={styles.label}>Upload Bloodwork:</label>
          <input
            type="file"
            onChange={(e) => setBloodworkFile(e.target.files[0])}
            required
            style={styles.input}
          />
        </div>

        <button type="submit" style={styles.submitButton}>Submit</button>
      </form>
    </div>
  );
};

// Styling
const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    background: 'linear-gradient(to bottom, #cce6ff, white)',
    minHeight: '100vh',
    padding: '20px',
  },
  heading: {
    color: '#005b96',
    marginBottom: '20px',
    fontSize: '2em',
  },
  symptomsContainer: {
    display: 'flex',
    flexWrap: 'wrap',
    justifyContent: 'center',
    marginBottom: '20px',
  },
  symptomButton: {
    padding: '10px 15px',
    margin: '5px',
    borderRadius: '5px',
    border: 'none',
    cursor: 'pointer',
    color: '#005b96',
    fontSize: '1em',
    transition: 'background-color 0.3s',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    width: '100%',
    maxWidth: '400px',
    backgroundColor: '#e6f2ff',
    padding: '20px',
    borderRadius: '10px',
    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
  },
  inputGroup: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    width: '100%',
    marginBottom: '15px',
  },
  label: {
    display: 'block',
    marginBottom: '5px',
    color: '#005b96',
    fontSize: '1em',
  },
  input: {
    width: '100%',
    padding: '10px',
    borderRadius: '5px',
    border: '1px solid #b3d1ff',
    boxSizing: 'border-box',
    margin: '20px',
  },
  submitButton: {
    padding: '10px 20px',
    backgroundColor: '#005b96',
    color: 'white',
    border: 'none',
    borderRadius: '5px',
    cursor: 'pointer',
    fontSize: '1em',
    transition: 'background-color 0.3s',
  },
};

export default SymptomsForm;
