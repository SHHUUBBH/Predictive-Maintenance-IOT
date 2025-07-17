
# Predictive Maintenance Dashboard

This dashboard provides a visual interface for monitoring equipment health and predicting failures.

## Features

- Real-time monitoring of sensor data
- Anomaly detection and visualization
- Performance metrics for the predictive model
- Historical data analysis

## Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the dashboard:
   ```
   streamlit run app.py
   ```

3. Open your browser and navigate to http://localhost:8501

## Usage

- Use the sidebar to filter data by date range and select sensors to display
- Adjust the anomaly detection threshold to control sensitivity
- View real-time sensor readings and detected anomalies
- Monitor model performance metrics

## Customization

To customize the dashboard for your specific use case:

1. Modify the `load_data()` function to load your actual data
2. Update the `load_model()` function to use your trained model
3. Adjust the visualizations and metrics as needed
