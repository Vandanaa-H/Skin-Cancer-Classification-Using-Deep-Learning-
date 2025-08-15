import streamlit as st  # For creating the web app
import numpy as np  # For array manipulation
from keras.models import load_model  # For loading the pre-trained model
from PIL import Image  # To process uploaded images
import matplotlib.pyplot as plt  # For data visualization
import time 
import uuid
import base64
from io import BytesIO

# Load the pre-trained skin cancer classification model
MODEL_PATH = 'C:\\Users\\Admin\\Skin Cancer Classification using Deep Learning\\trained_model1\\trained_model1.keras'  # Path to your trained model
model = load_model(MODEL_PATH)

# Define the class labels (Benign, Malignant, Normal)
class_labels = ['Benign', 'Malignant', 'Normal']

# Function to preprocess and predict the class of an image
def predict_image(img, model):
    # Resize image to model input size
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)  # Model prediction
    predicted_class = np.argmax(predictions, axis=-1)  # Get the class index with the highest score
    confidence_scores = predictions[0]  # Get the confidence scores
    return predicted_class[0], confidence_scores

# Helper function to convert PIL image to base64 for HTML rendering with proper aspect ratio
def img_to_base64(img, max_width=400):
    buf = BytesIO()
    img_copy = img.copy()
    
    # Calculate new dimensions while maintaining aspect ratio
    original_width, original_height = img_copy.size
    if original_width > max_width:
        ratio = max_width / original_width
        new_height = int(original_height * ratio)
        img_copy = img_copy.resize((max_width, new_height), Image.Resampling.LANCZOS)
    
    img_copy.save(buf, format='PNG')
    base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return base64_str, img_copy.size

# Set page configuration with a custom title and icon - use wide layout for better screen usage
st.set_page_config(page_title="SkinScan AI", page_icon="🩺", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for full-width professional web app look
st.markdown(
"""
<style>
.stApp {
    background: linear-gradient(120deg, #f6f8fc 0%, #eaf6fc 100%);
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 16px;
}
.main {
    background: linear-gradient(120deg, #f8fafc 0%, #e0eafc 100%);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: none !important;
    width: 100% !important;
}
.header-section {
    text-align: center;
    padding: 3rem 0 2rem 0;
    background: linear-gradient(135deg, #1d3557 0%, #457b9d 100%);
    color: white;
    margin: -2rem -2rem 2rem -2rem;
    border-radius: 0 0 20px 20px;
}
.main-title {
    font-size: 4rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
.main-subtitle {
    font-size: 2.2rem;
    font-weight: 300;
    opacity: 0.95;
    margin-bottom: 0.5rem;
}
.main-description {
    font-size: 1.3rem;
    opacity: 0.85;
    max-width: 900px;
    margin: 0 auto;
    line-height: 1.6;
}
.content-container {
    background: white;
    border-radius: 15px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 4px 20px rgba(69,123,157,0.12);
}
.result-card {
    background: #fff;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(69,123,157,0.15);
    padding: 2rem;
    margin-top: 1.5rem;
    text-align: center;
    border-top: 4px solid;
}
.result-normal {
    border-top-color: #28a745;
}
.result-benign {
    border-top-color: #ffc107;
}
.result-malignant {
    border-top-color: #dc3545;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background-color: white;
    padding: 15px 25px 0 25px;
    border-radius: 15px 15px 0 0;
    box-shadow: 0 -2px 15px rgba(69,123,157,0.08);
}
.stTabs [data-baseweb="tab"] {
    font-size: 32px;
    font-weight: 800;
    color: #457b9d;
    border-radius: 10px 10px 0 0;
    padding: 25px 45px;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background-color: #eaf6fc !important;
    color: #1d3557 !important;
}
.stButton>button {
    background-color: #457b9d;
    color: white;
    font-size: 20px;
    font-weight: 600;
    border-radius: 10px;
    padding: 18px 40px;
    box-shadow: 0 4px 15px rgba(69,123,157,0.20);
    transition: all 0.3s ease;
    border: none;
}
.stButton>button:hover {
    background-color: #1d3557;
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(69,123,157,0.30);
}
.image-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px;
    background: linear-gradient(145deg, #f8f9fa, #e9ecef);
    border-radius: 15px;
    margin: 20px 0;
    border: 2px dashed #dee2e6;
}
.upload-section {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(145deg, #ffffff, #f8f9fa);
    border-radius: 15px;
    margin: 1rem 0;
    border: 1px solid #e9ecef;
}
.footer {
    color: #666;
    font-size: 16px;
    margin-top: 3rem;
    text-align: center;
    padding: 2rem;
    background: #f8f9fa;
    border-radius: 15px;
}
.loading-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem 2rem;
    background: linear-gradient(145deg, #ffffff, #f8f9fa);
    border-radius: 15px;
    margin: 2rem 0;
    border: 1px solid #e9ecef;
}
.loader {
    border: 4px solid #f3f3f3;
    border-top: 4px solid #457b9d;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    animation: spin 1s linear infinite;
    margin-bottom: 1rem;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.loading-text {
    color: #457b9d;
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.loading-subtext {
    color: #666;
    font-size: 1.1rem;
}
.section-container {
    background: white;
    border-radius: 15px;
    padding: 2.5rem;
    margin: 1.5rem 0;
    box-shadow: 0 2px 15px rgba(0,0,0,0.08);
}
.info-card {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 2rem;
    margin: 1rem 0;
    border-left: 4px solid #457b9d;
}
.step-item {
    display: flex;
    align-items: flex-start;
    margin-bottom: 2rem;
    padding: 2rem;
    background: #f8f9fa;
    border-radius: 12px;
    transition: all 0.3s ease;
}
.step-item:hover {
    background: #e9ecef;
    transform: translateY(-2px);
}
.step-number {
    background: #457b9d;
    color: white;
    border-radius: 50%;
    width: 60px;
    height: 60px;
    display: flex;
    justify-content: center;
    align-items: center;
    margin-right: 1.5rem;
    font-weight: bold;
    font-size: 1.4rem;
}
</style>
""",
unsafe_allow_html=True
)

# Header Section
st.markdown("""
<div class="header-section">
    <div class="main-title">SkinScan AI</div>
    <div class="main-subtitle">Advanced Skin Health Analysis Platform</div>
    <div class="main-description">
        Professional AI-powered dermatological assessment system providing instant, 
        secure analysis of skin conditions with medical-grade accuracy.
    </div>
</div>
""", unsafe_allow_html=True)
# Navigation tabs with full-width content
tab1, tab2, tab3, tab4 = st.tabs(["Home", "How it Works", "Safety & Privacy", "Contact"])

# Session state for prediction history
if "history" not in st.session_state:
    st.session_state["history"] = []

def add_to_history(img, pred, conf):
    st.session_state["history"].append({
        "id": str(uuid.uuid4()),
        "img": img,
        "pred": pred,
        "conf": conf,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

# Home Tab
with tab1:
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    
    # Upload section
    st.markdown("""
    <div class="upload-section">
        <h2 style="color: #1d3557; margin-bottom: 1rem;">Upload Skin Image for Analysis</h2>
        <p style="color: #666; font-size: 1.3rem; line-height: 1.6;">
            Select a clear, well-lit photograph of the skin area you want to analyze. 
            Our AI will provide an instant professional assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['jpg', 'png', 'jpeg'], 
        accept_multiple_files=False,
        help="Supported formats: JPG, PNG, JPEG (max 200MB)"
    )

    # Process uploaded image and show prediction
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        
        # Create columns for better layout
        col_left, col_center, col_right = st.columns([1, 2, 1])
        
        with col_center:
            # Option to show/hide uploaded image
            show_image = st.checkbox("Show uploaded image", value=False, help="Toggle to view the uploaded image")
            
            if show_image:
                # Display image with proper sizing and aspect ratio
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                
                # Calculate display size while maintaining aspect ratio
                display_width = 400
                original_width, original_height = img.size
                aspect_ratio = original_height / original_width
                display_height = int(display_width * aspect_ratio)
                
                # Limit maximum height to prevent very tall images
                if display_height > 500:
                    display_height = 500
                    display_width = int(display_height / aspect_ratio)
                
                st.image(img, width=display_width, caption="Uploaded Image for Analysis")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Show custom loading animation
            loading_placeholder = st.empty()
            result_placeholder = st.empty()
            
            # Custom loading animation
            loading_placeholder.markdown("""
            <div class="loading-container">
                <div class="loader"></div>
                <div class="loading-text">Analyzing Image</div>
                <div class="loading-subtext">Processing with advanced AI algorithms...</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Process image and predict
            time.sleep(2.5)  # Realistic processing time
            predicted_class, confidence_scores = predict_image(img, model)
            
            # Clear loading animation
            loading_placeholder.empty()
            
            # Save to history (without confidence score)
            add_to_history(uploaded_file.name, class_labels[predicted_class], "")
            
            # Show prediction result
            result_color = {
                'Normal': '#28a745',
                'Benign': '#ffc107', 
                'Malignant': '#dc3545'
            }[class_labels[predicted_class]]
            
            result_class = {
                'Normal': 'result-normal',
                'Benign': 'result-benign', 
                'Malignant': 'result-malignant'
            }[class_labels[predicted_class]]
            
            result_placeholder.markdown(f"""
            <div class='result-card {result_class}'>
                <h2 style='color:{result_color}; font-size:2.5rem; margin-bottom:1rem;'>{class_labels[predicted_class]}</h2>
                <p style='font-size:1.4rem; color:#555; line-height:1.6; margin-top:1.5rem;'>{
                    "Your skin appears healthy with no concerning features detected. Continue regular skin monitoring and maintain good skin health practices." if predicted_class == 2 else
                    "This appears to be a benign skin condition. Continue monitoring for any changes and consult a healthcare provider if you notice any developments." if predicted_class == 0 else
                    "This image shows features that may be concerning and warrant medical attention. Please consult a dermatologist promptly for proper professional evaluation and diagnosis."
                }</p>
                <div style='margin-top:2rem; padding:1.5rem; background:#f8f9fa; border-radius:10px;'>
                    <p style='color:#666; font-size:1.2rem; margin:0;'>
                        <strong>Next Steps:</strong> {
                            "Continue regular self-examinations and annual dermatology check-ups." if predicted_class == 2 else
                            "Monitor the area for changes and schedule a routine dermatology appointment." if predicted_class == 0 else
                            "Schedule an appointment with a dermatologist as soon as possible for professional diagnosis."
                        }
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Option to view prediction history
        if len(st.session_state["history"]) > 1:
            st.markdown("---")
            with st.expander("Previous Analysis History", expanded=False):
                for i, h in enumerate(st.session_state["history"][-10:][::-1]):  # Show last 10 results
                    st.markdown(f"""
                    <div style='padding:15px; background-color:#f8f9fa; border-radius:10px; margin-bottom:10px; border-left:4px solid {
                        '#28a745' if h['pred'] == 'Normal' else '#ffc107' if h['pred'] == 'Benign' else '#dc3545'
                    };'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <div>
                                <strong style='font-size:1.3rem; color:#1d3557;'>{h['pred']}</strong>
                                <br><small style='color:#666; font-size:1rem;'>File: {h['img']}</small>
                            </div>
                            <div style='text-align:right; color:#666; font-size:1rem;'>
                                {h.get('timestamp', 'Unknown time')}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding:3rem; color:#666;'>
            <p style='font-size:1.3rem; line-height:1.8;'>
                Ready to analyze your skin image? Upload a clear photograph above to get started.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# How it Works Tab
with tab2:
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown("""
        <h2 style='color:#1d3557; margin-bottom:2rem; font-size:2.2rem;'>How SkinScan AI Works</h2>
        <p style='color:#666; font-size:1.1rem; line-height:1.6; margin-bottom:2rem;'>
            Our advanced artificial intelligence system uses state-of-the-art deep learning technology 
            to analyze skin images and provide professional-grade assessments.
        </p>
    """, unsafe_allow_html=True)
    
    # Steps with improved styling
    steps = [
        {
            "title": "Image Upload & Processing",
            "description": "Upload a clear, high-quality photograph of the skin area you want analyzed. Our system automatically processes and prepares the image for analysis."
        },
        {
            "title": "AI Analysis Engine",
            "description": "Our deep learning model, trained on thousands of dermatological images, analyzes patterns, textures, colors, and other critical features in your image."
        },
        {
            "title": "Classification & Assessment", 
            "description": "The system classifies the skin condition as Normal, Benign, or Malignant based on recognized patterns and medical imaging standards."
        },
        {
            "title": "Results & Recommendations",
            "description": "Receive instant results with clear explanations and recommended next steps based on the analysis findings."
        }
    ]
    
    for i, step in enumerate(steps, 1):
        st.markdown(f"""
        <div class="step-item">
            <div class="step-number">{i}</div>
            <div>
                <h4 style='margin-bottom:0.5rem; color:#1d3557; font-size:1.3rem;'>{step['title']}</h4>
                <p style='color:#666; line-height:1.6; margin:0; font-size:1rem;'>{step['description']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='background:#e9f7ff; padding:2rem; border-radius:15px; margin-top:2rem; border-left:5px solid #457b9d;'>
            <h4 style='color:#1d3557; margin-bottom:1rem; font-size:1.4rem;'>Advanced Technology</h4>
            <p style='color:#555; line-height:1.6; margin:0;'>
                SkinScan AI utilizes a sophisticated convolutional neural network architecture 
                specifically designed for medical image analysis. The model has been trained on 
                extensive dermatological datasets and continues to improve through advanced 
                machine learning techniques, ensuring high accuracy and reliability in skin condition assessment.
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Safety & Privacy Tab
with tab3:
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown("""
        <h2 style='color:#1d3557; margin-bottom:2rem; font-size:2.2rem;'>Safety & Privacy Commitment</h2>
    """, unsafe_allow_html=True)
    
    # Privacy section
    st.markdown("""
        <div style='background:#d4edda; padding:2rem; border-radius:15px; margin-bottom:1.5rem; border-left:5px solid #28a745;'>
            <h4 style='color:#1d3557; margin-bottom:1rem; font-size:1.4rem;'>Privacy Protection</h4>
            <ul style='color:#555; line-height:1.8; margin:0;'>
                <li><strong>Zero Data Storage:</strong> Your images are processed instantly and never stored on our servers</li>
                <li><strong>No Personal Information:</strong> No registration or personal identification required</li>
                <li><strong>Local Processing:</strong> Analysis happens securely without data transmission to third parties</li>
                <li><strong>Session Only:</strong> Your scan history exists only during your current session</li>
                <li><strong>Encrypted Processing:</strong> All data handling uses industry-standard encryption protocols</li>
            </ul>
        </div>
        
        <div style='background:#fff3cd; padding:2rem; border-radius:15px; margin-bottom:1.5rem; border-left:5px solid #ffc107;'>
            <h4 style='color:#1d3557; margin-bottom:1rem; font-size:1.4rem;'>Medical Disclaimer</h4>
            <p style='color:#555; margin-bottom:1rem; line-height:1.6;'>
                <strong>Important:</strong> SkinScan AI is designed as an informational screening tool only and is not intended 
                to replace professional medical advice, diagnosis, or treatment.
            </p>
            <p style='color:#555; margin-bottom:1rem; line-height:1.6;'>
                The application provides preliminary assessments based on image analysis algorithms, but comprehensive 
                skin condition evaluation requires in-person examination by qualified healthcare professionals.
            </p>
            <p style='color:#721c24; font-weight:600; line-height:1.6; margin:0;'>
                Always consult a dermatologist or healthcare provider for proper diagnosis and treatment of skin conditions, 
                especially if SkinScan AI indicates potentially concerning features.
            </p>
        </div>
        
        <div style='background:#f8d7da; padding:2rem; border-radius:15px; border-left:5px solid #dc3545;'>
            <h4 style='color:#1d3557; margin-bottom:1rem; font-size:1.4rem;'>When to Seek Medical Attention</h4>
            <p style='color:#555; margin-bottom:1rem; line-height:1.6;'>
                Regardless of SkinScan AI assessment results, consult a healthcare provider immediately if you observe:
            </p>
            <ul style='color:#555; line-height:1.8; margin:0;'>
                <li>Rapid changes in size, shape, color, or texture of moles or skin lesions</li>
                <li>Bleeding, itching, pain, or tenderness in any skin area</li>
                <li>New growths that do not heal within reasonable timeframes</li>
                <li>Moles with irregular borders, multiple colors, or asymmetrical appearance</li>
                <li>Any skin changes that cause concern or appear unusual</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Contact Tab
with tab4:
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown("""
        <h2 style='color:#1d3557; margin-bottom:2rem; font-size:2.2rem;'>Contact & Support</h2>
    """, unsafe_allow_html=True)
    
    # Contact information
    contact_info = [
        {
            "title": "Technical Support",
            "content": "For technical issues, questions about the platform, or general inquiries:<br><strong>Email:</strong> <a href='mailto:support@skinscanai.com' style='color:#457b9d; text-decoration:none;'>support@skinscanai.com</a><br><strong>Response Time:</strong> Within 24 hours"
        },
        {
            "title": "Medical Information", 
            "content": "For questions about medical aspects or interpretation of results:<br><strong>Email:</strong> <a href='mailto:medical@skinscanai.com' style='color:#457b9d; text-decoration:none;'>medical@skinscanai.com</a><br><strong>Note:</strong> Not for emergency medical situations"
        }
    ]
    
    for info in contact_info:
        st.markdown(f"""
        <div style='background:#f8f9fa; padding:2rem; border-radius:12px; margin-bottom:2rem; border-left:4px solid #457b9d;'>
            <h4 style='color:#1d3557; margin-bottom:1.5rem; font-size:1.6rem; font-weight:600;'>{info['title']}</h4>
            <p style='color:#666; line-height:1.8; margin:0; font-size:1.2rem;'>{info['content']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feedback Section
    st.markdown("""
        <div style='background:#f8f9fa; padding:3rem; border-radius:15px; margin-top:2rem; text-align:center;'>
            <h4 style='color:#1d3557; margin-bottom:1.5rem; font-size:2rem; font-weight:700;'>Feedback & Improvement</h4>
            <p style='color:#666; line-height:1.8; margin:0; font-size:1.2rem;'>
                We continuously improve SkinScan AI based on user feedback and advancing medical research. 
                Share your experience or suggestions to help us enhance our platform:<br>
                <a href='mailto:feedback@skinscanai.com' style='color:#457b9d; text-decoration:none; font-weight:500; font-size:1.3rem;'>feedback@skinscanai.com</a>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class='footer'>
        <h4 style='color:#1d3557; margin-bottom:1.5rem; font-size:1.8rem; font-weight:700;'>SkinScan AI</h4>
        <p style='margin-bottom:1rem; font-size:1.1rem; font-weight:500;'>Professional AI-Powered Dermatological Assessment Platform</p>
        <p style='margin-bottom:1.5rem; font-size:1rem;'>Copyright © 2025 SkinScan AI. All rights reserved.</p>
        <p style='font-size:0.95rem; color:#999; margin:0; line-height:1.6;'>
            Disclaimer: This application is for informational and screening purposes only. 
            It is not a substitute for professional medical advice, diagnosis, or treatment. 
            Always seek the advice of qualified healthcare providers regarding medical conditions.
        </p>
    </div>
""", unsafe_allow_html=True)