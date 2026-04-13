# 🐾 PetPulse AI: Cat & Dog Classifier

PetPulse AI is a high-performance image classification engine designed to distinguish between cats and dogs with precision. Utilizing a sophisticated machine learning pipeline and a sleek, modern interface, it provides instant and reliable pet identification.

## 🚀 Features
- **Intelligent Classification**: Powered by a Neural Network with Multi-feature Fusion (HOG + LBP + Color Histograms).
- **Modern Dashboard**: A premium, responsive web interface for seamless image uploads and results.
- **Micro-animations**: Smooth transitions and interactive elements for an enhanced UX.
- **Robust Pipeline**: Automated image preprocessing and feature extraction.

## 🛠️ Technology Stack
- **Backend**: Python, Flask
- **Machine Learning**: Scikit-Learn, OpenCV, NumPy
- **Frontend**: HTML5, Vanilla CSS3 (Modern Glassmorphism Design)

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd "Assignment 1"
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

## 🧠 Model Details
The classifier employs a **Fusion Engine** that combines:
- **HOG (Histogram of Oriented Gradients)**: For structural and shape information.
- **LBP (Local Binary Patterns)**: For texture analysis.
- **Color Histograms**: For color distribution features.

These features are fed into a **Neural Network (MLP)** for final classification, ensuring high accuracy and reliability.

---
*Developed for Excellence in AI.*
