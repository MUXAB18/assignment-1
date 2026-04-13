import numpy as np
import cv2
import os
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog, local_binary_pattern
import pickle
import time
import random

# PetPulse AI Ultra-Extreme Configuration
IMG_SIZE = 128
HOG_PARAMS = {'orientations': 9, 'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2), 'block_norm': 'L2-Hys', 'transform_sqrt': True}
LBP_PARAMS = {'P': 8, 'R': 1, 'method': 'uniform'}
LIMIT = 10000  # Extreme Scale

def extract_features(img_color):
    try:
        img_color = cv2.resize(img_color, (IMG_SIZE, IMG_SIZE))
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        # 1. HOG
        feat_hog = hog(img_gray, **HOG_PARAMS)
        # 2. LBP
        lbp = local_binary_pattern(img_gray, LBP_PARAMS['P'], LBP_PARAMS['R'], LBP_PARAMS['method'])
        (feat_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_PARAMS['P'] + 3), range=(0, LBP_PARAMS['P'] + 2))
        feat_lbp = feat_lbp.astype("float"); feat_lbp /= (feat_lbp.sum() + 1e-6)
        # 3. COLOR
        h_b = cv2.calcHist([img_color], [0], None, [32], [0, 256]).flatten()
        h_g = cv2.calcHist([img_color], [1], None, [32], [0, 256]).flatten()
        h_r = cv2.calcHist([img_color], [2], None, [32], [0, 256]).flatten()
        feat_color = np.concatenate([h_b, h_g, h_r]); feat_color /= (feat_color.sum() + 1e-6)
        
        return np.concatenate([feat_hog, feat_lbp, feat_color])
    except: return None

def get_augmented_features(img_path):
    img = cv2.imread(img_path)
    if img is None: return []
    
    feats = []
    # Original
    f1 = extract_features(img)
    if f1 is not None: feats.append(f1)
    
    # Mirror
    f2 = extract_features(cv2.flip(img, 1))
    if f2 is not None: feats.append(f2)
    
    # Rotation (5 degrees)
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 5, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    f3 = extract_features(rotated)
    if f3 is not None: feats.append(f3)
    
    return feats

def train_ultra_extreme():
    start_time = time.time()
    data_dir = os.path.expanduser("~/Desktop/data")
    X, y = [], []
    classes = {'cats': 0, 'dogs': 1}
    
    print("\n[1/4] ULTRA EXTRACTION: Crunching 25,000 images with Triple Augmentation...")
    for class_name, label in classes.items():
        folder = os.path.join(data_dir, "train", class_name)
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.seed(42); random.shuffle(files)
        
        print(f"  Processing {min(LIMIT, len(files))} images for {class_name}...")
        for i in range(min(LIMIT, len(files))):
            aug_feats = get_augmented_features(os.path.join(folder, files[i]))
            for f in aug_feats:
                X.append(f)
                y.append(label)
            if (i+1) % 1000 == 0:
                print(f"    Progress: {i+1} images ({len(X)} feature vectors total)")

    X_train, y_train = np.array(X), np.array(y)
    
    print("\n[2/4] SCALING: Normalizing Ultra-Extreme Feature Matrix...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    print("\n[3/4] ENSEMBLING: Training the High-Council of AI Models...")
    # 1. Neural Network
    mlp = MLPClassifier(hidden_layer_sizes=(512, 128), max_iter=300, random_state=42)
    # 2. Calibrated SVM
    svm = CalibratedClassifierCV(LinearSVC(C=1.0, dual=False, random_state=42), cv=3)
    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('mlp', mlp), ('svm', svm), ('rf', rf)],
        voting='soft'
    )
    
    ensemble.fit(X_train, y_train)
    
    print("\n[4/4] SAVING: Hard-locking labels and exporting ensembled brain...")
    # Verified labels: Cat is definitely Index 0 for this build
    pipeline = {
        'model': ensemble,
        'scaler': scaler,
        'img_size': IMG_SIZE,
        'hog_params': HOG_PARAMS,
        'lbp_params': LBP_PARAMS,
        'classes': ["Cat", "Dog"],
        'version': '3.5-ULTRA-EXTREME'
    }
    
    with open('cat_dog_hog_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
        
    print(f"\nSUCCESS: Ultra-Extreme Engine deployed in {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    train_ultra_extreme()
