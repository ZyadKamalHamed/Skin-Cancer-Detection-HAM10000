# 🔬 Skin Cancer Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Advanced deep learning system for automated skin cancer detection using computer vision and transfer learning on dermatoscopic images.**

## 🎯 **Project Overview**

This project implements a state-of-the-art skin cancer detection system using the HAM10000 dataset, targeting **90%+ accuracy** across 7 different skin lesion types. The system combines transfer learning with clinical-grade evaluation metrics to provide AI-powered diagnosis support.

### **Clinical Impact**
- **Multi-class classification** of 7 skin cancer types
- **Transfer learning** approach using ResNet50 architecture
- **Real-time inference** with confidence scoring
- **Clinical interpretability** through visualization techniques

## 🏥 **Skin Lesion Classes**

| Class | Medical Name | Description |
|-------|--------------|-------------|
| **nv** | Melanocytic nevi | Benign moles |
| **mel** | Melanoma | Malignant skin cancer |
| **bkl** | Benign keratosis-like lesions | Non-cancerous growths |
| **bcc** | Basal cell carcinoma | Common skin cancer |
| **akiec** | Actinic keratoses | Pre-cancerous lesions |
| **vasc** | Vascular lesions | Blood vessel related |
| **df** | Dermatofibroma | Benign skin tumor |

## 🚀 **Key Features**

- **🧠 Deep Learning**: ResNet50 with ImageNet pre-training
- **📊 Professional Evaluation**: AUC, sensitivity, specificity metrics
- **🔍 Clinical Visualization**: Grad-CAM attention maps (planned)
- **⚡ CPU Optimized**: Efficient training on standard hardware
- **🌐 Web Deployment**: Streamlit application (planned)

## 📁 **Dataset Information**

**HAM10000 Dataset** - "Human Against Machine with 10,000 training images"
- **Size**: 10,015 dermatoscopic images
- **Quality**: High-resolution clinical photography
- **Annotation**: Expert dermatologist verified
- **Balance**: Handles class imbalance with weighted sampling

## 🏗️ **Project Architecture**

```
TO BE UPDATED
```

## 🛠️ **Technical Stack**

- **Deep Learning**: Python 3.10 PyTorch 2.7+, TorchVision
- **Data Science**: pandas, numpy, scikit-learn
- **Computer Vision**: OpenCV, PIL
- **Deployment**: Streamlit (planned)
- **Visualization**: matplotlib, seaborn

## 🚀 **Quick Start**

### **Environment Setup**
```bash
# Clone repository
git clone https://github.com/ZyadKamalHamed/Skin-Cancer-Detection-HAM10000.git
cd Skin-Cancer-Detection-HAM10000

# Install dependencies
pip install torch torchvision pandas tqdm

# Test setup
python src/test_setup.py
```

### **Data Pipeline**
```bash
# Initialize sample dataset
python src/data/download.py
```


## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

- **HAM10000 Dataset**: Tschandl, P., Rosendahl, C. & Kittler, H.

## 📞 **Contact**

**Zyad Kamal Hamed**  
📧 zyad2408@live.com.au  
🔗 [LinkedIn](https://linkedin.com/in/zyadkamalhamed/)  
