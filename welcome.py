import streamlit as st
import numpy as np
import pickle

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD THE SAVED MODEL ---
try:
    with open('model.pkl', 'rb') as file:
        svmc = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found. Please ensure it's in the same directory.")
    st.stop()


# --- STYLING (CSS) ---
st.markdown("""
    <style>
    /* Import Google Font & Font Awesome */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

    /* General styling */
    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Titles and Headers */
    h1, h2, h3 {
        color: #d81b60;
        text-align: center;
        font-weight: 700;
    }

    /* Custom containers/cards for content */
    .custom-card {
        padding: 30px;
        border-radius: 15px;
        background-color: var(--secondary-background-color);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        color: var(--text-color);
        border: 1px solid var(--gray-200);
    }
    
    .custom-card p, .custom-card li, .custom-card b, .custom-card strong, .custom-card ol {
        color: var(--text-color);
    }

    /* Feature Cards on Welcome Page */
    .feature-card {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s;
        color: var(--text-color);
        border: 1px solid var(--gray-200);
        height: 100%; /* Ensures cards in a row are the same height */
    }
    .feature-card p {
        color: var(--text-color);
        opacity: 0.8;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .feature-card .icon {
        font-size: 40px;
        color: #d81b60;
        margin-bottom: 10px;
    }
    
    /* Disclaimer Card on About Page */
    .disclaimer-card {
        padding: 25px;
        border-radius: 15px;
        background-color: #fffbe6; /* Light yellow */
        border-left: 10px solid #ffc107; /* Amber */
        margin-top: 20px;
        color: #5d4037; /* Brownish text for readability */
    }
    .disclaimer-card h3 {
        color: #ff8f00; /* Darker Amber for header */
    }
    .disclaimer-card p, .disclaimer-card strong {
         color: #5d4037;
    }


    /* Prediction Button */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(45deg, #d81b60, #ff4081);
        color: white;
        font-size: 20px;
        font-weight: 600;
        padding: 15px;
        border: none;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease-in-out;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0px 6px 12px rgba(216, 27, 96, 0.4);
    }

    /* Prediction Result Cards */
    .result-card {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        border-left: 10px solid;
    }
    .result-card .icon { font-size: 50px; margin-bottom: 15px; }
    .result-card h3 { margin-top: 0; font-size: 28px; }
    .result-card-benign { background-color: #e8f5e9; border-color: #4caf50; }
    .result-card-benign .icon, .result-card-benign h3, .result-card-benign p { color: #2e7d32; }
    .result-card-malignant { background-color: #ffebee; border-color: #f44336; }
    .result-card-malignant .icon, .result-card-malignant h3, .result-card-malignant p { color: #c62828; }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        font-size: 14px;
        color: var(--text-color);
        opacity: 0.7;
    }
    </style>
""", unsafe_allow_html=True)


# --- PREDICTION RESULT FUNCTION ---
def show_prediction_result(prediction_val):
    if prediction_val == 1:
        st.markdown(
            '<div class="result-card result-card-malignant">'
            '<div class="icon"><i class="fa-solid fa-triangle-exclamation"></i></div>'
            '<h3>Prediction: Malignant</h3>'
            '<p>The model predicts a high probability of malignancy. It is crucial to consult a healthcare professional for a definitive diagnosis and guidance.</p>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-card result-card-benign">'
            '<div class="icon"><i class="fa-solid fa-circle-check"></i></div>'
            '<h3>Prediction: Benign</h3>'
            '<p>The model predicts a low probability of malignancy. Regular check-ups are still highly recommended for continued monitoring.</p>'
            '</div>',
            unsafe_allow_html=True
        )

# --- PAGE 1: WELCOME PAGE ---
def welcome_page():
    # st.image("https://images.unsplash.com/photo-1581091226825-a6a2a5a0a4da?q=80&w=2070&auto=format&fit=crop", use_container_width=True)
    st.title("🎗️ Breast Cancer Diagnosis Predictor")
    
    st.markdown("""
    <div class="custom-card">
        Welcome! This tool uses a <strong>Support Vector Machine (SVM)</strong> model to provide an early prediction on breast cancer diagnosis based on tumor features. 
        Early detection is paramount, and this app aims to be a supportive first step in the diagnostic process.
        <br><br>
        Navigate to the <b>Prediction</b> page to begin.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Key Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="icon"><i class="fa-solid fa-robot"></i></div>
            <h4>AI-Powered</h4>
            <p>Leverages a robust machine learning model with ~98% accuracy for reliable predictions.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="icon"><i class="fa-solid fa-bolt"></i></div>
            <h4>Instant Results</h4>
            <p>Input the diagnostic data and receive an immediate prediction without any delay.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="icon"><i class="fa-solid fa-shield-halved"></i></div>
            <h4>Secure & Private</h4>
            <p>Your data is processed in real-time and is never stored, ensuring complete privacy.</p>
        </div>
        """, unsafe_allow_html=True)


# --- PAGE 2: PREDICTION PAGE ---
def prediction_page():
    st.title("🧪 Prediction Form")
    st.markdown("Enter the diagnostic measurements below. The data is grouped into three categories for ease of use.")

    mean_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
    se_features = ['radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se']
    worst_features = ['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
    
    entries = []

    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["📊 Mean Values", "📈 Standard Error", "📉 Worst Values"])

        def create_inputs(features, container):
            local_entries = []
            col1, col2 = container.columns(2)
            for i, label in enumerate(features):
                target_col = col1 if i % 2 == 0 else col2
                entry = target_col.number_input(f"{label}", value=0.0, format="%.4f", key=label)
                local_entries.append(entry)
            return local_entries

        with tab1:
            entries.extend(create_inputs(mean_features, st))
        with tab2:
            entries.extend(create_inputs(se_features, st))
        with tab3:
            entries.extend(create_inputs(worst_features, st))
        
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    if st.button("🔍 Predict Diagnosis"):
        with st.spinner('🧠 The AI is thinking...'):
            input_data = np.array(entries).reshape(1, -1)
            prediction_val = svmc.predict(input_data)
            show_prediction_result(prediction_val[0])

# --- PAGE 3: ABOUT PAGE ---
def about_page():
    st.title("ℹ️ About This Project")

    # --- Project Mission ---
    st.markdown("""
    <div class="custom-card">
        <h3>🎯 Project Mission</h3>
        <p>Our mission is to harness the power of machine learning to provide an accessible and intuitive tool that supports the crucial goal of early breast cancer detection. This application is designed for educational and preliminary informational purposes, aiming to bridge the gap between complex medical data and understandable insights.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- How to Use ---
    st.markdown("""
    <div class="custom-card">
        <h3>❓ How to Use This Tool</h3>
        <ol>
            <li>Navigate to the <strong>🧪 Prediction</strong> page from the sidebar.</li>
            <li>Carefully enter the values for the 30 cytopathological features as obtained from a medical report.</li>
            <li>Click the <strong>"Predict Diagnosis"</strong> button to receive the model's prediction.</li>
        </ol>
        <p>The result will indicate whether the features correspond more closely to a benign or malignant diagnosis based on the patterns the model has learned from the data.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- The Technology Expander ---
    with st.expander("🤖 Click to learn about the Technology Behind the Predictor"):
        st.subheader("The Model: Support Vector Machine (SVM)")
        st.write(
            """
            This project uses a Support Vector Machine, a highly effective supervised machine learning algorithm. 
            Think of it as an expert drawing the most precise boundary possible to separate two groups—in this case, 
            separating the complex characteristics of benign and malignant tumors in a multi-dimensional space.
            """
        )
        
        st.subheader("The Dataset")
        st.write(
            """
            The model was trained on the renowned **Breast Cancer Wisconsin (Diagnostic) Dataset** from the 
            University of Wisconsin Hospitals, Madison. This dataset is a classic in the machine learning 
            community and contains 569 instances with 30 different features, such as:
            """
        )
        st.markdown(
            """
            - Tumor radius and perimeter
            - Texture (standard deviation of gray-scale values)
            - Smoothness and compactness
            """
        )

    # --- Disclaimer ---
    st.markdown("""
    <div class="disclaimer-card">
        <h3>⚠️ Important Disclaimer</h3>
        <p>This tool is for <strong>informational and educational purposes only</strong>. It is not a substitute for professional medical advice, diagnosis, or treatment. The predictions made by this AI model are based on data patterns and are not a definitive medical conclusion.</p>
        <p><strong>Always consult a qualified healthcare provider</strong> with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read or seen on this application.</p>
    </div>
    """, unsafe_allow_html=True)


# --- MAIN APP ---
def main():
    # --- UPDATED: Removed image and added emoji to title ---
    st.sidebar.title("🎗️ Navigation")
    
    page_options = {
        "🏠 Welcome": welcome_page,
        "🧪 Prediction": prediction_page,
        "ℹ️ About": about_page
    }
    
    page_selection = st.sidebar.radio("Go to", list(page_options.keys()))
    page_options[page_selection]()

    st.markdown('<div class="footer">🌸 Developed with care for Breast Cancer Awareness</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()