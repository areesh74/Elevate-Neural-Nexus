import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import joblib
from elasticsearch import Elasticsearch
import plotly.graph_objects as go
import random

# Initialize Elasticsearch client
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])  # Add 'scheme' parameter

# Load Employee Data
@st.cache
def load_data():
    data = pd.read_csv('employeePerformance.csv')
    return data

# Preprocess and Train Model
def train_model(data):
    target = 'Performance_Score'
    features = data.drop(['Employee_ID', 'Performance_Score', 'Resigned', 'Hire_Date'], axis=1)
    target_data = data[target]

    # Encode categorical features
    categorical_cols = features.select_dtypes(include=['object']).columns
    features = pd.get_dummies(features, columns=categorical_cols, drop_first=True)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(features, target_data, test_size=0.3, random_state=42)

    # Standardize numeric data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train CatBoost Model
    model = CatBoostRegressor(verbose=0)
    model.fit(X_train_scaled, y_train)

    # Save model and scaler
    joblib.dump(model, 'catboost_performance_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    return model, scaler, X_train, y_train

# Prediction Function (with controlled fluctuation)
def predict_performance(input_data, model, scaler, feature_columns, es_results=None):
    # Convert input to DataFrame for scaling
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    # React to Elasticsearch results
    if es_results and len(es_results['hits']['hits']) > 0:
        scam_factor = len(es_results['hits']['hits']) * 0.1  # Example adjustment
        prediction = max(prediction - scam_factor, 40)  # Minimum score of 40

    # Introduce controlled randomness
    random.seed(hash(tuple(input_data.items())) % (10**6))  # Seed based on input
    fluctuation = random.uniform(-15, 15)  # Random fluctuation between -15 and 15
    fluctuated_prediction = min(max(prediction + fluctuation, 40), 85)  # Clamp between 40 and 85

    return fluctuated_prediction

# Filtering Function
def filter_employees(data, filters):
    filtered_data = data.copy()
    for col, val in filters.items():
        if val:
            filtered_data = filtered_data[filtered_data[col] == val]
    return filtered_data

# Generate Growth Message
def generate_growth_message(score):
    if score >= 90:
        return "Outstanding performance! Keep up the excellent work and consider mentoring peers."
    elif score >= 75:
        return "Great job! Focus on consistent improvements and take up challenging projects."
    elif score >= 60:
        return "Good work! Enhance your skills through additional training and workshops."
    elif score >= 40:
        return "Fair performance. Identify key areas for improvement and seek feedback."
    else:
        return "Needs improvement. Collaborate with your manager to create a development plan."

# Visualize Predictions
def visualize_prediction(score):
    # Categories and dynamic value distribution
    categories = ['Needs Improvement', 'Fair', 'Good', 'Great', 'Outstanding']
    values = [
        max(5, 40 - score) if score < 40 else 5,
        max(5, score - 40) if 40 <= score < 60 else 5,
        max(5, score - 60) if 60 <= score < 75 else 5,
        max(5, score - 75) if 75 <= score < 90 else 5,
        max(5, score - 90) if score >= 90 else 5,
    ]

    fig_pie = go.Figure(data=[go.Pie(labels=categories, values=values, hole=0.4)])
    fig_pie.update_layout(title="Performance Categorization")

    # Bar chart for actual vs. predicted performance
    fig_bar = go.Figure(data=[
        go.Bar(name="Predicted Score", x=["Employee"], y=[score]),
        go.Bar(name="Ideal Score", x=["Employee"], y=[100]),
    ])
    fig_bar.update_layout(
        title="Predicted Performance vs. Ideal",
        xaxis_title="Performance",
        yaxis_title="Score",
        barmode="group",
    )
    return fig_pie, fig_bar

# Load Dataset
data = load_data()

# Train Model if not already trained
if 'catboost_performance_model.pkl' not in st.session_state:
    model, scaler, X_train, y_train = train_model(data)
    st.session_state['model'] = model
    st.session_state['scaler'] = scaler
    st.session_state['feature_columns'] = X_train.columns.tolist()

# Streamlit App Layout
st.title("Employee Performance Prediction and Filtering")

# Filtering Section
st.sidebar.header("Employee Filters")
filters = {
    'Department': st.sidebar.selectbox('Filter by Department', options=[None] + list(data['Department'].unique())),
    'Gender': st.sidebar.selectbox('Filter by Gender', options=[None] + list(data['Gender'].unique())),
    'Job_Title': st.sidebar.selectbox('Filter by Job Title', options=[None] + list(data['Job_Title'].unique())),
    'Education_Level': st.sidebar.selectbox('Filter by Education Level', options=[None] + list(data['Education_Level'].unique())),
}

filtered_data = filter_employees(data, filters)
st.write(f"Filtered Employees: {len(filtered_data)}")
st.dataframe(filtered_data)

# Elasticsearch Section
st.sidebar.header("Elasticsearch Query")
es_query = st.sidebar.text_input("Search Employees (e.g., Job_Title: 'Analyst')")

if st.sidebar.button("Search in Elasticsearch"):
    es_query_body = {"query": {"query_string": {"query": es_query}}}
    es_results = es.search(index="employees", body=es_query_body)
    st.write("Search Results:")
    for hit in es_results['hits']['hits']:
        st.json(hit['_source'])
else:
    es_results = None

# Performance Prediction Section
st.sidebar.header("Employee Features for Prediction")
employee_input = {
    'Department': st.sidebar.selectbox('Department', data['Department'].unique()),
    'Gender': st.sidebar.selectbox('Gender', data['Gender'].unique()),
    'Age': st.sidebar.slider('Age', 18, 60, 30),
    'Job_Title': st.sidebar.selectbox('Job Title', data['Job_Title'].unique()),
    'Years_At_Company': st.sidebar.slider('Years at Company', 0, 40, 5),
    'Education_Level': st.sidebar.selectbox('Education Level', data['Education_Level'].unique()),
    'Monthly_Salary': st.sidebar.slider('Monthly Salary', 2000, 20000, 5000),
    'Work_Hours_Per_Week': st.sidebar.slider('Work Hours Per Week', 20, 60, 40),
    'Projects_Handled': st.sidebar.slider('Projects Handled', 0, 100, 5),
    'Overtime_Hours': st.sidebar.slider('Overtime Hours', 0, 50, 5),
    'Sick_Days': st.sidebar.slider('Sick Days', 0, 20, 2),
    'Remote_Work_Frequency': st.sidebar.slider('Remote Work Frequency', 0, 100, 50),
    'Team_Size': st.sidebar.slider('Team Size', 1, 50, 10),
    'Training_Hours': st.sidebar.slider('Training Hours', 0, 100, 20),
    'Promotions': st.sidebar.slider('Promotions', 0, 10, 1),
    'Employee_Satisfaction_Score': st.sidebar.slider('Employee Satisfaction Score', 0.0, 5.0, 2.5),
}

if st.button("Predict Performance"):
    model = st.session_state['model']
    scaler = st.session_state['scaler']
    feature_columns = st.session_state['feature_columns']

    prediction = predict_performance(employee_input, model, scaler, feature_columns, es_results)
    st.success(f"The predicted performance score is: {prediction:.2f}")

    # Generate growth message
    growth_message = generate_growth_message(prediction)
    st.info(growth_message)

    # Plotly visualizations
    fig_pie, fig_bar = visualize_prediction(prediction)
    st.plotly_chart(fig_pie)
    st.plotly_chart(fig_bar)

