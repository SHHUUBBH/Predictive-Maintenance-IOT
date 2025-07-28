import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Model, Scaler, AND Feature Names ---
try:
    model = joblib.load('./model_assets/predictive_maintenance_model.joblib')
    scaler = joblib.load('./model_assets/scaler.joblib')
    model_features = joblib.load('./model_assets/feature_names.joblib')
except FileNotFoundError:
    st.error("Model, scaler, or feature_names file not found. Please run the main training script first to generate these files.")
    st.stop()

# --- App Layout ---
st.set_page_config(page_title="Predictive Maintenance Report", layout="wide")
st.title("⚙️ Comprehensive Predictive Maintenance Report")
st.write(
    "This report automatically analyzes the entire dataset and predicts potential equipment failures using the trained model."
)

# --- Load and Process Data ---
try:
    # Paths updated to reflect the folder structure
    train_file_path = '../aps_failure_at_scania_trucks/aps_failure_training_set.csv'
    test_file_path = '../aps_failure_at_scania_trucks/aps_failure_test_set.csv'
    
    df_train = pd.read_csv(train_file_path)
    df_test = pd.read_csv(test_file_path)
    df_full_dataset = pd.concat([df_train, df_test], ignore_index=True)

    with st.spinner("Processing data and making predictions..."):
        df_processed = df_full_dataset.replace('na', np.nan)
        df_processed = df_processed[model_features]
        df_processed = df_processed.apply(pd.to_numeric)
        
        imputer = SimpleImputer(strategy='median')
        df_imputed = pd.DataFrame(imputer.fit_transform(df_processed), columns=model_features)
        
        df_scaled = scaler.transform(df_imputed)
        
        predictions = model.predict(df_scaled)
        probabilities = model.predict_proba(df_scaled)[:, 1]

except Exception as e:
    st.error(f"An error occurred during data loading or processing: {e}")
    st.stop()


# --- Display Results ---
results_df = pd.DataFrame({
    'Component_ID': df_full_dataset.index,
    'Prediction': ['Failure' if p == 1 else 'Normal' for p in predictions],
    'Failure_Probability': probabilities
})
failures = results_df[results_df['Prediction'] == 'Failure']


# --- Main Visualization Section ---
st.write("### 📊 Prediction Summary")

col1, col2 = st.columns([1, 2]) # Create two columns for summary

with col1:
    # Display summary text
    st.metric(label="Total Components Analyzed", value=f"{len(results_df):,}")
    st.metric(label="Predicted Failures", value=f"{len(failures):,}", delta_color="inverse")
    
    if len(failures) > 0:
        st.warning(f"Found {len(failures)} component(s) at risk.")
    else:
        st.success("No components predicted to fail.")

with col2:
    # --- Prediction Summary Pie Chart ---
    prediction_counts = results_df['Prediction'].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.pie(prediction_counts, labels=prediction_counts.index, autopct='%1.1f%%', 
           startangle=90, colors=['skyblue', 'salmon'],
           wedgeprops={"edgecolor": "white", 'linewidth': 1})
    ax.set_title("Proportion of Predictions", fontweight='bold')
    st.pyplot(fig)


# --- Detailed Analysis Section ---
if not failures.empty:
    st.write("---")
    st.write("### 🔬 Detailed Failure Analysis")

    col3, col4 = st.columns(2)

    with col3:
        # --- Feature Importance Plot ---
        st.write("**Top 15 Most Important Sensors for Prediction**")
        feature_importances = pd.DataFrame({
            'feature': model_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='importance', y='feature', data=feature_importances, ax=ax, palette='viridis')
        ax.set_title('Feature Importance', fontweight='bold')
        st.pyplot(fig)

    with col4:
        # --- Failure Probability Distribution ---
        st.write("**Distribution of Failure Probabilities**")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(failures['Failure_Probability'], kde=True, ax=ax, color='salmon', bins=20)
        ax.set_title('Model Confidence for Failure Predictions', fontweight='bold')
        ax.set_xlabel('Predicted Probability of Failure')
        st.pyplot(fig)

    # --- Table of Failing Components ---
    st.write("---")
    st.write("### 📋 Components Predicted to Fail")
    st.dataframe(
        failures.style.format({'Failure_Probability': '{:.2%}'})
        .background_gradient(subset=['Failure_Probability'], cmap='Reds')
    )