# Skin Cancer Classification using Deep Learning

##  Project Overview  
This project implements an **AI-powered skin cancer classification system** using deep learning to classify skin lesions into **Benign**, **Malignant**, and **Normal** categories. The project features a custom CNN model trained from scratch and a professional-grade web application for real-time skin analysis.

###  Key Features
- **Advanced CNN Model**: Custom-built deep learning model achieving ~86% accuracy
- **Professional Web Interface**: Streamlit-based application with medical-grade UI/UX
- **Real-time Analysis**: Instant skin lesion classification with professional feedback
- **Secure Processing**: Zero-storage architecture ensuring complete privacy
- **Medical Compliance**: Professional disclaimer and safety guidelines
- **Responsive Design**: Full-width web application optimized for all devices

---

##  Project Structure  

```
Skin Cancer Classification using Deep Learning/
│
├──  skin_cancer_dataset/          # Organized dataset directory
│   ├──  train/                    # Training dataset (80%)
│   │   ├──  benign/               # Non-cancerous lesions
│   │   ├──  malignant/            # Cancerous lesions
│   │   └──  normal/               # Normal skin samples
│   └──  test/                     # Testing dataset (20%)
│       ├──  benign/
│       ├──  malignant/
│       └──  normal/
│
├──  trained_model1/               # Model artifacts
│   ├──  trained_model1.keras     # Saved trained model
│   └──  __results___files/       # Training visualizations
│
├──  app.py                       # Professional Streamlit web application
├──  SkinCancerClassification.ipynb  # Model training & evaluation notebook
├──  CONTEXT.md                   # Project documentation
├──  README.md                    # GitHub documentation
└──  requirements.txt             # Python dependencies
```

---

##  Dataset Information  

### Data Sources
1. **Malignant & Benign Lesions**:  
   [Kaggle Dataset – Skin Cancer: Malignant vs Benign](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)  

2. **Normal Skin Samples**:  
   [Kaggle Dataset – Normal Skin Images](https://www.kaggle.com/datasets/ahdasdwdasd/our-normal-skin)  

### Data Preprocessing
- **Image Standardization**: Resized to 224×224 pixels
- **Pixel Normalization**: Values scaled to [0, 1] range
- **Data Augmentation**: 
  - Rotation: ±20 degrees
  - Width/Height Shift: ±20%
  - Shear Transformation: ±20%
  - Zoom Range: ±20%
  - Horizontal Flip: Enabled

---

##  Model Architecture  

### Custom CNN Design
```python
Sequential([
    Conv2D(4, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    
    Conv2D(8, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: Benign, Malignant, Normal
])
```

### Training Configuration
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 50
- **Batch Size**: 32
- **Validation Split**: 20%

---

##  Model Performance  

### Evaluation Metrics
- **Overall Accuracy**: ~86%
- **Training Strategy**: Early stopping to prevent overfitting
- **Validation**: Comprehensive testing on unseen data

### Visualizations Generated
- Training vs Validation Accuracy curves
- Training vs Validation Loss curves
- Confusion Matrix analysis
- Classification Report (Precision, Recall, F1-Score)

---

##  Web Application Features  

### SkinScan AI Interface
The Streamlit application (`app.py`) provides a professional medical-grade interface with:

#### **Professional Design**
- Medical-themed color scheme (#1d3557, #457b9d, #e63946)
- Gradient backgrounds and professional styling
- Full-width responsive layout
- Custom CSS for professional appearance

#### **Multi-Tab Navigation**
1. **Home**: Image upload and analysis interface
2. **How it Works**: Detailed explanation of the AI process
3. **Safety & Privacy**: Data security and medical disclaimers
4. **Contact**: Support information and feedback system

#### **Security & Privacy**
- Zero-storage architecture
- Local image processing
- No data transmission to external servers
- Session-only data retention

#### **User Experience**
- Drag-and-drop file upload
- Real-time loading animations
- Professional result cards with color-coded classifications
- Conditional image display option
- Analysis history tracking (session-based)

#### **Medical Compliance**
- Comprehensive medical disclaimers
- Professional healthcare guidance
- Clear limitations and recommendations
- Emergency consultation guidelines

---

##  Technology Stack  

### Core Technologies
- **Python 3.11+**: Primary programming language
- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **NumPy**: Numerical computing
- **Pillow (PIL)**: Image processing
- **Matplotlib**: Data visualization

### Additional Libraries
- **scikit-learn**: Machine learning utilities
- **pandas**: Data manipulation
- **seaborn**: Statistical visualization
- **base64**: Image encoding for web display

---

##  Getting Started  

### Prerequisites
```bash
Python 3.11 or higher
pip package manager
```

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/skin-cancer-classification.git
   cd skin-cancer-classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the application**:
   Open your browser and navigate to `http://localhost:8501`

---

##  Usage Instructions  

### For Model Training
1. Open `SkinCancerClassification.ipynb` in Jupyter Notebook
2. Ensure dataset is properly organized in `skin_cancer_dataset/`
3. Run all cells to train the model
4. Model will be saved as `trained_model1.keras`

### For Web Application
1. Launch the Streamlit app using `streamlit run app.py`
2. Upload a clear, well-lit skin image
3. Click "Analyze Image" for instant AI classification
4. Review results and professional recommendations
5. Consult healthcare providers for medical decisions

---

## Important Disclaimers  

### Medical Disclaimer
- This application is for **informational and screening purposes only**
- **Not a substitute** for professional medical advice, diagnosis, or treatment
- Always consult qualified healthcare providers for medical decisions
- Seek immediate medical attention for concerning skin changes

### AI Limitations
- Model accuracy: ~86% (not 100% reliable)
- Designed as a screening tool, not diagnostic instrument
- Results should be validated by medical professionals
- Emergency situations require immediate medical consultation

---

## License  
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing  
Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

---

** Always prioritize professional medical consultation over AI analysis results.**

