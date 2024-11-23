import streamlit as st
import numpy as np
import pickle

# Load the saved SVM model
with open('model.pkl', 'rb') as file:
    svmc = pickle.load(file)

# Define a function for the prediction
def predict(input_data):
    # Make a prediction
    prediction = svmc.predict(input_data)

    # Display the prediction
    if prediction[0] == 1:
        prediction_value= 'Malignant'
    else:
        prediction_value= "Benign"
    st.write("The predicted diagnosis is ",prediction_value)

# Welcome page function
def welcome():
    st.title("Welcome to Breast Cancer Prediction Website")
    st.write("""
    This website helps in predicting the likelihood of breast cancer based on several input parameters.
    
    Breast cancer is one of the most common cancers among women worldwide. Early detection and timely treatment significantly increase the chances of successful recovery. This website utilizes a machine learning model trained on a dataset of breast cancer features to provide predictions.

    To get started, please enter the required parameters in the sidebar and click the "Predict" button. You will receive a prediction regarding the likelihood of breast cancer based on the input values.

    Feel free to explore the website and learn more about breast cancer prediction.
    """)
    
    # Add an image
    st.image("image.jpg", caption="Breast Cancer Awareness", use_column_width=True)

# Prediction page function
def prediction():
    st.title("Cancer Prediction")
    # Add input fields for main parameters
    labels = ['Radius_mean', 'Texture_mean', 'Perimeter_mean', 'Area_mean', 'Smoothness_mean',
              'Compactness_mean', 'Concavity_mean', 'Concave points_mean', 'Symmetry_mean',
              'Fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se',
              'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
              'concave points_se', 'symmetry_se', 'fractal_dimension_se',
              'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
              'smoothness_worst', 'compactness_worst', 'concavity_worst',
              'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
    entries = []
    for i, label in enumerate(labels):
        entry = st.number_input(label, value=0.0)
        entries.append(entry)

    # Add a predict button
    if st.button("Predict"):
        # Get user inputs
        input_data = np.array(entries).reshape(1, -1)
        predict(input_data)
        
# About page function
def about():
    st.title("About")
    st.write("""
    This website uses a machine learning model to predict the likelihood of breast cancer based on input parameters such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

    The machine learning model is trained on a dataset known as the Breast Cancer Wisconsin (Diagnostic) Data Set, which contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The model aims to assist in early detection of breast cancer, which is crucial for effective treatment and improved survival rates.

    The model used in this website is a Support Vector Machine (SVM) classifier, which is a powerful tool for classification tasks. It has been trained on a labeled dataset to predict whether a given set of features corresponds to a benign or malignant breast mass.

    It's important to note that while this website provides predictions based on statistical analysis, it should not be used as a substitute for professional medical advice. Always consult with a healthcare professional for accurate diagnosis and treatment options.

    We hope this website serves as a helpful tool in raising awareness about breast cancer and the importance of early detection. Thank you for using our website.
    """)

# Create the main function
def main():
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Go to", ("Welcome", "Prediction", "About"))

    if page == "Welcome":
        welcome()

    elif page == "Prediction":
        prediction()


    elif page == "About":
        about()

if __name__ == "__main__":
    main()



