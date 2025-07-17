import pandas as pd
import numpy as np
import pickle
import os
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the preprocessing components
with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)
    imputer = preprocessor['imputer']
    scaler = preprocessor['scaler']
    feature_names = preprocessor['feature_names']

# Create the Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the request data
        request_data = request.get_json()

        # Extract the feature values
        input_data = request_data['data']

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure all features are present
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = np.nan

        # Select only the required features in the correct order
        input_df = input_df[feature_names]

        # Preprocess the data
        input_df = input_df.replace('na', np.nan)
        input_df = input_df.apply(pd.to_numeric, errors='coerce')
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        # Make prediction
        prediction = int(model.predict(input_scaled)[0])
        probability = float(model.predict_proba(input_scaled)[0][1])

        # Return the prediction
        return jsonify({
            'prediction': prediction,
            'probability': probability,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy'
    })

@app.route('/', methods=['GET'])
def home():
    return """
    <html>
        <head>
            <title>Predictive Maintenance API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                h2 { color: #666; }
                pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Predictive Maintenance API</h1>
            <p>This API provides endpoints for predicting equipment failures.</p>

            <h2>Endpoints</h2>

            <h3>POST /predict</h3>
            <p>Make a prediction based on sensor data.</p>
            <p>Example request:</p>
            <pre>
{
    "data": {
        "feature_0": 0.5,
        "feature_1": -1.2,
        ...
    }
}
            </pre>
            <p>Example response:</p>
            <pre>
{
    "prediction": 0,
    "probability": 0.12345,
    "status": "success"
}
            </pre>

            <h3>GET /health</h3>
            <p>Check if the API is healthy.</p>
            <p>Example response:</p>
            <pre>
{
    "status": "healthy"
}
            </pre>
        </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
