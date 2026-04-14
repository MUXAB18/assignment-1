from flask import Flask, render_template, request, jsonify
import pickle
import cv2
import numpy as np
import os
from skimage.feature import hog, local_binary_pattern

app = Flask(__name__)

# Constants
PIPELINE_PATH = 'cat_dog_hog_pipeline.pkl'
FALLBACK_PIPELINE = 'cat_dog_pipeline.pkl'

def load_pipeline():
    """Reload the pipeline from disk to get latest training results"""
    if os.path.exists(PIPELINE_PATH):
        with open(PIPELINE_PATH, 'rb') as f:
            return pickle.load(f)
    elif os.path.exists(FALLBACK_PIPELINE):
        with open(FALLBACK_PIPELINE, 'rb') as f:
            return pickle.load(f)
    return None

def extract_fusion_features(image_path, pipeline):
    """Extract Shape, Texture, and Color features matching the 3.0 Fusion Engine"""
    try:
        img_size = pipeline.get('img_size', 128)
        hog_params = pipeline.get('hog_params', {})
        lbp_params = pipeline.get('lbp_params', {'P': 8, 'R': 1, 'method': 'uniform'})
        
        # Load Color and Grayscale
        img_color = cv2.imread(image_path)
        if img_color is None: return None
        img_color = cv2.resize(img_color, (img_size, img_size))
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        # 1. SHAPE: HOG
        feat_hog = hog(img_gray, **hog_params)
        
        # 2. TEXTURE: LBP
        lbp = local_binary_pattern(img_gray, lbp_params['P'], lbp_params['R'], lbp_params['method'])
        (feat_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, lbp_params['P'] + 3), range=(0, lbp_params['P'] + 2))
        feat_lbp = feat_lbp.astype("float")
        feat_lbp /= (feat_lbp.sum() + 1e-6)
        
        # 3. COLOR: BGR Histograms
        hist_b = cv2.calcHist([img_color], [0], None, [32], [0, 256]).flatten()
        hist_g = cv2.calcHist([img_color], [1], None, [32], [0, 256]).flatten()
        hist_r = cv2.calcHist([img_color], [2], None, [32], [0, 256]).flatten()
        feat_color = np.concatenate([hist_b, hist_g, hist_r])
        feat_color /= (feat_color.sum() + 1e-6)
        
        return np.concatenate([feat_hog, feat_lbp, feat_color])
    except Exception as e:
        print(f"Feature Extraction Error: {e}")
        return None

def predict_image(image_path):
    """Predict Cat/Dog using the latest Fusion Engine pipeline"""
    pipeline = load_pipeline()
    if pipeline is None:
        return None, None, None
        
    model = pipeline['model']
    scaler = pipeline.get('scaler') # Fusion model uses a scaler
    
    # Extract
    features = extract_fusion_features(image_path, pipeline)
    if features is None:
        return None, None, None
    
    # Scale if needed
    if scaler:
        features = scaler.transform(features.reshape(1, -1))
    else:
        features = features.reshape(1, -1)
    
    # Predict
    probabilities = model.predict_proba(features)[0]
    
    # Safe Mapping using pipeline metadata
    label_names = pipeline.get('classes', ['Cat', 'Dog'])
    results = {label_names[i]: probabilities[i] for i in range(len(probabilities))}
    
    cat_prob = results.get('Cat', 0)
    dog_prob = results.get('Dog', 0)
    
    prediction = "Cat" if cat_prob > dog_prob else "Dog"
    
    return prediction, cat_prob * 100, dog_prob * 100

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)
    
    prediction, cat_prob, dog_prob = predict_image(filepath)
    
    if prediction is None:
        return jsonify({'error': 'Model not loaded or feature extraction failed.'})
    
    return jsonify({
        'prediction': prediction,
        'cat_probability': f"{cat_prob:.2f}%",
        'dog_probability': f"{dog_prob:.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5002)
