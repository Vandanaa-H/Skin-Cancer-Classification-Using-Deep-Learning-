#  Skin Cancer Classification using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Professional AI-powered dermatological assessment platform featuring SkinScan AI web interface**

![Skin Cancer Classification Banner](https://via.placeholder.com/800x200/1d3557/ffffff?text=Skin+Cancer+Classification+using+Deep+Learning)

##  Project Overview

This project implements a comprehensive deep learning solution for skin cancer classification that categorizes skin lesions into **Benign**, **Malignant**, and **Normal** categories using a custom-built Convolutional Neural Network. The project features SkinScan AI - a professional medical-grade web interface built with Streamlit, providing real-time skin analysis with a focus on user privacy and medical compliance.

###  Key Highlights
-  **Custom CNN Model** - Built from scratch achieving high accuracy
-  **SkinScan AI Interface** - Professional medical-grade web application
-  **Zero-Storage Architecture** - Complete privacy with local processing
-  **Medical Compliance** - Professional disclaimers and safety guidelines
-  **Responsive Design** - Optimized for all devices and screen sizes

##  Live Demo

Try the application: [SkinScan AI Demo](your-deployment-url-here)

##  Screenshots

<table>
  <tr>
    <td><img src="https://via.placeholder.com/400x250/e9f7ff/1d3557?text=Home+Interface" alt="Home Interface" width="400"/></td>
    <td><img src="https://via.placeholder.com/400x250/f8f9fa/457b9d?text=Analysis+Results" alt="Analysis Results" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>Professional Home Interface</b></td>
    <td align="center"><b>AI Analysis Results</b></td>
  </tr>
</table>

##  Installation & Setup

### Prerequisites
- Python 3.11 or higher
- pip package manager
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/skin-cancer-classification.git
   cd skin-cancer-classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - Upload a skin image and get instant AI analysis

##  Model Performance

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | ~86% |
| **Precision** | High |
| **Recall** | High |
| **F1-Score** | Balanced |

### Model Architecture
```
Input (224x224x3)
    ↓
Conv2D(4) + MaxPool → Conv2D(8) + MaxPool → Conv2D(16) + MaxPool → Conv2D(32) + MaxPool
    ↓
Flatten → Dense(128) → Dense(3, softmax)
    ↓
Output: [Benign, Malignant, Normal]
```

##  Web Application Features

### **Multi-Tab Interface**
- **Home**: Upload and analyze skin images
- **How it Works**: Detailed AI process explanation
- **Safety & Privacy**: Data security information
- **Contact**: Support and feedback system

###  **Privacy & Security**
- Zero-storage architecture
- Local image processing
- No external data transmission
- Session-only data retention

###  **User Experience**
- Drag-and-drop file upload
- Real-time loading animations
- Professional result cards
- Color-coded classifications
- Analysis history tracking

##  Project Structure

```
Skin Cancer Classification using Deep Learning/
│
├── 📁 skin_cancer_dataset/          # Dataset directory
│   ├── 📁 train/                    # Training data (80%)
│   │   ├── 📁 benign/               # Non-cancerous lesions
│   │   ├── 📁 malignant/            # Cancerous lesions
│   │   └── 📁 normal/               # Normal skin samples
│   └── 📁 test/                     # Testing data (20%)
│       ├── 📁 benign/
│       ├── 📁 malignant/
│       └── 📁 normal/
│
├── 📁 trained_model1/               # Model artifacts
│   ├── 📄 trained_model1.keras     # Saved trained model
│   └── 📁 __results___files/       # Training visualizations
│
├── 📄 app.py                       # Streamlit web application
├── 📄 SkinCancerClassification.ipynb  # Training notebook
├── 📄 requirements.txt             # Dependencies
├── 📄 README.md                    # This file
└── 📄 CONTEXT.md                   # Detailed documentation
```

##  Dataset Information

### Data Sources
- **Malignant & Benign**: [Kaggle - Skin Cancer Dataset](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)
- **Normal Skin**: [Kaggle - Normal Skin Images](https://www.kaggle.com/datasets/ahdasdwdasd/our-normal-skin)

### Data Preprocessing
- **Image Resize**: 224×224 pixels
- **Normalization**: Pixel values [0, 1]
- **Augmentation**: Rotation, shift, shear, zoom, flip

##  Tech Stack

<table>
  <tr>
    <td align="center"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></td>
    <td align="center"><img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/></td>
    <td align="center"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/></td>
  </tr>
  <tr>
    <td align="center"><img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/></td>
    <td align="center"><img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/></td>
    <td align="center"><img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib"/></td>
  </tr>
</table>

##  Usage Guide

### For Model Training
1. Open `SkinCancerClassification.ipynb` in Jupyter Notebook
2. Ensure dataset is organized in `skin_cancer_dataset/`
3. Run all cells to train the model
4. Model saves automatically as `trained_model1.keras`

### For Web Application
1. Launch: `streamlit run app.py`
2. Upload a clear, well-lit skin image
3. Click "Analyze Image" for instant classification
4. Review results and recommendations
5. **Always consult healthcare providers for medical decisions**

##  Important Medical Disclaimer

> ** MEDICAL DISCLAIMER**
> 
> This application is for **informational and screening purposes only**. It is **NOT a substitute** for professional medical advice, diagnosis, or treatment. 
> 
> **Always consult qualified healthcare providers** regarding medical conditions and seek immediate medical attention for concerning skin changes.

##  Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Support & Contact

###  Technical Support
- **Email**: support@skinscanai.com
- **Response Time**: Within 24 hours

###  Feedback & Suggestions
- **Email**: feedback@skinscanai.com
- Help us improve the platform!

###  Medical Inquiries
- **Email**: medical@skinscanai.com
- **Note**: Not for emergency medical situations

##  Acknowledgments

- Dataset providers from Kaggle community
- TensorFlow and Keras development teams
- Streamlit for the amazing web framework
- Medical community for guidance on compliance

##  Future Enhancements

- [ ] Mobile application development
- [ ] Advanced model architectures (ResNet, EfficientNet)
- [ ] Multi-language support
- [ ] Integration with medical databases
- [ ] Batch processing capabilities
- [ ] Enhanced data augmentation techniques

---

<div align="center">

** Always prioritize professional medical consultation over AI analysis results.**

</div>
