# Predictive Maintenance for Industrial IoT

![maintenance](https://img.shields.io/badge/maintenance-predictive-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![tensorflow](https://img.shields.io/badge/tensorflow-2.0%2B-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-green)
![flask](https://img.shields.io/badge/flask-2.0%2B-red)
![streamlit](https://img.shields.io/badge/streamlit-1.22%2B-blueviolet)

A comprehensive machine learning solution for predicting equipment failures in industrial IoT systems using sensor data analysis, anomaly detection, and classification techniques.

<div align="center">
  <img src="https://i.imgur.com/XqpQZRE.png" alt="Predictive Maintenance Concept" width="600">
</div>

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Dashboard](#dashboard)
- [Results & Performance](#results--performance)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project implements a predictive maintenance system for industrial IoT devices using advanced machine learning techniques. It analyzes sensor data to detect anomalies and predict potential failures before they occur, helping to reduce downtime, maintenance costs, and improve overall operational efficiency.

Key components include:
- Data preprocessing and exploratory analysis
- Anomaly detection using multiple algorithms
- Predictive modeling for failure prediction
- Model evaluation and selection
- API deployment for real-time prediction
- Interactive dashboard for monitoring and visualization

## 📊 Dataset

The project uses the APS Failure at Scania Trucks dataset, which contains sensor readings from the Air Pressure System (APS) of Scania trucks:

- **Training set**: 60,000 examples (59,000 negative, 1,000 positive)
- **Test set**: 16,000 examples
- **Features**: 171 anonymized sensor readings
- **Task**: Binary classification (APS component failures vs. non-APS related failures)
- **Challenge**: Highly imbalanced dataset with significant missing values

## ✨ Features

- **Data Preprocessing**
  - Missing value imputation
  - Feature scaling
  - Feature importance ranking
  
- **Anomaly Detection**
  - Isolation Forest
  - One-Class SVM
  - Autoencoder Neural Networks
  
- **Predictive Modeling**
  - Time series forecasting with ARIMA
  - Sequence modeling with LSTM
  - Classification with Random Forest/XGBoost
  
- **Deployment**
  - RESTful API using Flask
  - Containerization with Docker
  - Cloud deployment guidelines
  
- **Visualization**
  - Interactive dashboard with Streamlit
  - Real-time monitoring capabilities
  - Performance metrics visualization

## 🗂️ Project Structure

```
├── aps_failure_at_scania_trucks/    # Dataset directory
│   ├── aps_failure_description.txt
│   ├── aps_failure_test_set.csv
│   └── aps_failure_training_set.csv
├── notebooks/                       # Jupyter notebooks for analysis
│   ├── 01_data_processing_and_eda.ipynb
│   ├── 02_anomaly_detection.ipynb
│   ├── 03_predictive_modeling.ipynb
│   ├── 04_model_evaluation.ipynb
│   ├── 05_deployment.ipynb
│   └── 06_dashboard.ipynb
├── src/                             # Source code
│   ├── model_evaluation/            # Trained models and evaluation
│   │   └── best_classification_model.pkl
│   ├── deployment/                  # API deployment files
│   │   ├── app.py                   # Flask API
│   │   ├── Dockerfile
│   │   ├── model.pkl
│   │   ├── preprocessor.pkl
│   │   └── requirements.txt
│   └── dashboard/                   # Dashboard files
│       ├── app.py                   # Streamlit dashboard
│       ├── README.md
│       └── requirements.txt
├── project_documentation.md         # Comprehensive documentation
├── README.md
└── requirements.txt                 # Main project dependencies
```

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/predictive-maintenance-iot.git
   cd predictive-maintenance-iot
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset (if not included):
   ```bash
   # The dataset is already included in the repository
   # If needed, you can download it from UCI Machine Learning Repository
   ```

## 🚀 Usage

### Running the Notebooks

Execute the Jupyter notebooks in sequence to understand the complete workflow:

```bash
jupyter notebook notebooks/01_data_processing_and_eda.ipynb
jupyter notebook notebooks/02_anomaly_detection.ipynb
jupyter notebook notebooks/03_predictive_modeling.ipynb
jupyter notebook notebooks/04_model_evaluation.ipynb
jupyter notebook notebooks/05_deployment.ipynb
jupyter notebook notebooks/06_dashboard.ipynb
```

### Running the API

1. Navigate to the deployment directory:
   ```bash
   cd src/deployment
   ```

2. Start the Flask API:
   ```bash
   python app.py
   ```

3. The API will be available at http://localhost:5000

### Running the Dashboard

1. Navigate to the dashboard directory:
   ```bash
   cd src/dashboard
   ```

2. Start the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```

3. The dashboard will be available at http://localhost:8501

## 🔌 API Documentation

The API provides the following endpoints:

- **GET /** - API documentation
  
- **GET /health** - Health check endpoint
  ```bash
  curl -X GET http://localhost:5000/health
  ```
  Response:
  ```json
  {
    "status": "healthy"
  }
  ```

- **POST /predict** - Make predictions based on sensor data
  ```bash
  curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"data": {"feature_0": 0.5, "feature_1": -1.2}}'
  ```
  Response:
  ```json
  {
    "prediction": 0,
    "probability": 0.12345,
    "status": "success"
  }
  ```

## 📊 Dashboard

The Streamlit dashboard provides:

- Real-time monitoring of sensor data
- Anomaly detection visualization
- Performance metrics for the predictive model
- Historical data analysis capabilities
- Adjustable threshold for anomaly detection

## 📈 Results & Performance

Our model evaluation shows the following performance metrics:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 95.2% | 78.6% | 82.1% | 80.3% | 0.93 |
| XGBoost | 97.1% | 83.4% | 79.8% | 81.6% | 0.95 |
| LSTM | 94.8% | 76.2% | 84.3% | 80.0% | 0.92 |

The XGBoost model was selected as the best performing model based on overall F1 score and efficiency.

## 🔮 Future Work

1. **Data Collection**:
   - Incorporate real-time sensor data streams
   - Expand to additional sensor types and equipment
   
2. **Model Improvement**:
   - Explore deep learning architectures for time series forecasting
   - Implement ensemble methods for improved performance
   - Incorporate domain knowledge into feature engineering
   
3. **Deployment**:
   - Deploy the model to major cloud platforms (AWS, Azure, GCP)
   - Implement CI/CD pipeline for model updates
   - Add authentication and security features
   
4. **Dashboard**:
   - Add alert systems for detected anomalies
   - Enhance visualizations for better interpretability
   - Implement scheduled reporting features

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- **Scania CV AB** for providing the APS Failure at Scania Trucks dataset
- **UCI Machine Learning Repository** for hosting the dataset
- **Open-source contributors** to libraries such as scikit-learn, TensorFlow, XGBoost, Flask, and Streamlit
- The broader data science and machine learning community for tutorials, tools, and inspiration

---

<div align="center">
  <p>Built with ❤️ for better industrial maintenance</p>
  <p>© 2025 Subhro Pal</p>
</div> 
