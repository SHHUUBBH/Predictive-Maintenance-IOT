import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import requests
import json
from datetime import datetime, timedelta
import time
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define functions for loading data and models
@st.cache_data
def load_data():
    # In a real application, you would load your actual data
    # For demonstration, we'll create synthetic data

    # Create a date range for the past 30 days
    dates = pd.date_range(end=datetime.now(), periods=30*24, freq='H')

    # Create sensor readings with some random noise
    data = pd.DataFrame({
        'timestamp': dates,
        'sensor_1': np.sin(np.linspace(0, 15*np.pi, len(dates))) + np.random.normal(0, 0.1, len(dates)),
        'sensor_2': np.cos(np.linspace(0, 15*np.pi, len(dates))) + np.random.normal(0, 0.1, len(dates)),
        'sensor_3': np.random.normal(0, 0.2, len(dates)).cumsum(),
        'sensor_4': np.random.normal(1, 0.1, len(dates)),
        'sensor_5': np.random.normal(0, 0.3, len(dates)).cumsum() + np.sin(np.linspace(0, 5*np.pi, len(dates)))
    })

    # Add some anomalies
    anomaly_indices = [100, 250, 400, 550, 650]
    for idx in anomaly_indices:
        data.loc[idx:idx+10, 'sensor_1'] += 2.0
        data.loc[idx:idx+10, 'sensor_2'] -= 1.5
        data.loc[idx:idx+10, 'sensor_4'] *= 1.5

    # Add a failure flag column (1 for failure, 0 for normal)
    data['failure'] = 0
    for idx in anomaly_indices:
        data.loc[idx+5:idx+15, 'failure'] = 1

    return data

@st.cache_resource
def load_model():
    # In a real application, you would load your trained model
    # For demonstration, we'll create a simple function that returns random predictions
    def dummy_model(data):
        # Calculate anomaly scores based on sensor readings
        scores = (
            np.abs(data['sensor_1']) + 
            np.abs(data['sensor_2']) + 
            np.abs(data['sensor_3']) + 
            np.abs(data['sensor_4']) + 
            np.abs(data['sensor_5'])
        ) / 5

        # Normalize scores to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min())

        # Return predictions and probabilities
        predictions = (scores > 0.7).astype(int)
        probabilities = scores

        return predictions, probabilities

    return dummy_model

# Load data and model
data = load_data()
model = load_model()

# Make predictions
predictions, probabilities = model(data)
data['prediction'] = predictions
data['probability'] = probabilities

# Calculate metrics
true_positives = ((data['prediction'] == 1) & (data['failure'] == 1)).sum()
false_positives = ((data['prediction'] == 1) & (data['failure'] == 0)).sum()
true_negatives = ((data['prediction'] == 0) & (data['failure'] == 0)).sum()
false_negatives = ((data['prediction'] == 0) & (data['failure'] == 1)).sum()

accuracy = (true_positives + true_negatives) / len(data)
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Create the Streamlit dashboard
st.title("Predictive Maintenance Dashboard")

# Sidebar
st.sidebar.header("Settings")
st.sidebar.subheader("Filter Data")

# Date range filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    [data['timestamp'].min().date(), data['timestamp'].max().date()]
)

# Convert date_range to datetime for filtering
start_date = pd.Timestamp(date_range[0])
end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

# Filter data based on date range
filtered_data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]

# Sensor selection
st.sidebar.subheader("Select Sensors")
selected_sensors = st.sidebar.multiselect(
    "Choose sensors to display",
    ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5'],
    default=['sensor_1', 'sensor_2']
)

# Threshold for anomaly detection
threshold = st.sidebar.slider("Anomaly Detection Threshold", 0.0, 1.0, 0.7, 0.01)

# Dashboard content
# Row 1: Key metrics
st.header("Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", f"{accuracy:.2%}")

with col2:
    st.metric("Precision", f"{precision:.2%}")

with col3:
    st.metric("Recall", f"{recall:.2%}")

with col4:
    st.metric("F1 Score", f"{f1:.2%}")

# Row 2: Sensor readings and anomaly detection
st.header("Sensor Readings and Anomaly Detection")

# Create a plotly figure
fig = go.Figure()

# Add sensor readings
for sensor in selected_sensors:
    fig.add_trace(go.Scatter(
        x=filtered_data['timestamp'],
        y=filtered_data[sensor],
        mode='lines',
        name=sensor
    ))

# Add anomaly markers
anomalies = filtered_data[filtered_data['probability'] > threshold]
if not anomalies.empty:
    fig.add_trace(go.Scatter(
        x=anomalies['timestamp'],
        y=[0] * len(anomalies),  # Place markers at the bottom
        mode='markers',
        marker=dict(
            symbol='x',
            color='red',
            size=10
        ),
        name='Detected Anomalies'
    ))

# Add true failures
failures = filtered_data[filtered_data['failure'] == 1]
if not failures.empty:
    fig.add_trace(go.Scatter(
        x=failures['timestamp'],
        y=[0] * len(failures),  # Place markers at the bottom
        mode='markers',
        marker=dict(
            symbol='circle',
            color='black',
            size=10
        ),
        name='True Failures'
    ))

# Update layout
fig.update_layout(
    title='Sensor Readings Over Time',
    xaxis_title='Time',
    yaxis_title='Value',
    height=500,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Row 3: Anomaly probability
st.header("Anomaly Probability")

# Create a plotly figure for anomaly probability
fig_prob = go.Figure()

# Add probability line
fig_prob.add_trace(go.Scatter(
    x=filtered_data['timestamp'],
    y=filtered_data['probability'],
    mode='lines',
    name='Anomaly Probability'
))

# Add threshold line
fig_prob.add_trace(go.Scatter(
    x=filtered_data['timestamp'],
    y=[threshold] * len(filtered_data),
    mode='lines',
    line=dict(color='red', dash='dash'),
    name=f'Threshold ({threshold})'
))

# Update layout
fig_prob.update_layout(
    title='Anomaly Probability Over Time',
    xaxis_title='Time',
    yaxis_title='Probability',
    height=400,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Display the plot
st.plotly_chart(fig_prob, use_container_width=True)

# Row 4: Recent anomalies table
st.header("Recent Anomalies")

# Get recent anomalies
recent_anomalies = filtered_data[filtered_data['probability'] > threshold].tail(10)
if not recent_anomalies.empty:
    # Format the table
    table_data = recent_anomalies[['timestamp', 'probability'] + selected_sensors].copy()
    table_data['timestamp'] = table_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    table_data['probability'] = table_data['probability'].map('{:.2%}'.format)

    # Display the table
    st.dataframe(table_data, use_container_width=True)
else:
    st.info("No anomalies detected in the selected time range.")

# Row 5: Confusion matrix
st.header("Model Performance")

col1, col2 = st.columns(2)

with col1:
    # Create confusion matrix
    cm = np.array([[true_negatives, false_positives], [false_negatives, true_positives]])

    # Create a plotly figure for confusion matrix
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Normal', 'Failure'],
        y=['Normal', 'Failure'],
        text_auto=True,
        color_continuous_scale='Blues'
    )

    # Update layout
    fig_cm.update_layout(
        title='Confusion Matrix',
        height=400
    )

    # Display the plot
    st.plotly_chart(fig_cm, use_container_width=True)

with col2:
    # Create ROC curve (simplified for demonstration)
    fpr = [0, false_positives / (false_positives + true_negatives), 1]
    tpr = [0, true_positives / (true_positives + false_negatives), 1]

    # Create a plotly figure for ROC curve
    fig_roc = go.Figure()

    # Add ROC curve
    fig_roc.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines+markers',
        name='ROC Curve'
    ))

    # Add diagonal line
    fig_roc.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Random'
    ))

    # Update layout
    fig_roc.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=400
    )

    # Display the plot
    st.plotly_chart(fig_roc, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Predictive Maintenance Dashboard | Created with Streamlit")
