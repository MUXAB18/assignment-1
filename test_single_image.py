import pickle
import cv2
import numpy as np
import os

# Configuration (fallback)
IMG_SIZE = 64

# Load the trained pipeline
PIPELINE_PATH = 'cat_dog_pipeline.pkl'
if os.path.exists(PIPELINE_PATH):
    with open(PIPELINE_PATH, 'rb') as f:
        pipeline = pickle.load(f)
        model = pipeline['model']
        pca = pipeline['pca']
        IMG_SIZE = pipeline.get('img_size', 64)
    print(f"Loaded pipeline from {PIPELINE_PATH}")
else:
    print(f"Error: {PIPELINE_PATH} not found.")
    exit()

def predict_single_image(image_path):
    """Predict Cat/Dog from a single image using the pipeline"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Preprocessing
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img_flat = img.flatten().reshape(1, -1)
    
    # Apply PCA
    img_pca = pca.transform(img_flat)
    
    # Predict
    probabilities = model.predict_proba(img_pca)[0]
    cat_prob = probabilities[0]
    dog_prob = probabilities[1]
    
    print(f"\nImage: {image_path}")
    print(f"Cat probability: {cat_prob * 100:.2f}%")
    print(f"Dog probability: {dog_prob * 100:.2f}%")
    
    if cat_prob > dog_prob:
        print(f"Prediction: CAT")
    else:
        print(f"Prediction: DOG")

# Test with an image
if __name__ == "__main__":
    # Put your image path here
    image_path = os.path.expanduser("~/Desktop/data/test/dogs/dog.100.jpg")
    if os.path.exists(image_path):
        predict_single_image(image_path)
    else:
        # Try to find any image in the test set
        test_dir = os.path.expanduser("~/Desktop/data/test/dogs")
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
            if files:
                predict_single_image(os.path.join(test_dir, files[0]))
            else:
                print("No images found in test directory.")
        else:
            print("Please set a valid image_path in the script.")
