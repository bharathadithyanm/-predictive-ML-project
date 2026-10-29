import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Page Title
st.title("🎓 Student Performance Predictor")

st.write("Enter student details below:")

# Inputs
hours = st.slider("Study Hours", 0, 10, 5)
attendance = st.slider("Attendance (%)", 0, 100, 70)
assignments = st.slider("Assignments Completed", 0, 10, 5)
midterm = st.slider("Midterm Marks", 0, 100, 50)
final = st.slider("Final Marks", 0, 100, 50)

# Load dataset
try:
    data = pd.read_csv("data/dataset.csv")
except:
    st.error("❌ dataset.csv not found. Check data folder.")
    st.stop()

# Features & Target
X = data.drop("target", axis=1)
y = data["target"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Predict button
if st.button("Predict Result"):
    input_data = pd.DataFrame([[hours, attendance, assignments, midterm, final]],
                              columns=X.columns)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Student will PASS")
    else:
        st.error("❌ Student may FAIL")