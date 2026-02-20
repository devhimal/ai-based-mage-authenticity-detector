# src/predict.py
import os
import argparse
import numpy as np
import cv2
import tensorflow as tf

# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def predict_single_image(image_path, model_path, img_size=(96, 96)):
    """
    Takes a path to a single image and classifies it as Real or Fake.
    - Loads the trained model.
    - Preprocesses the input image.
    - Predicts the class and prints the result with a confidence score.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please train the model first.")
        return

    try:
        # --- 1. Load Model ---
        print("Loading model...")
        model = tf.keras.models.load_model(model_path)

        # --- 2. Preprocess Image ---
        # Read the image, resize it, and normalize pixel values
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image file at {image_path}")
            return
            
        img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        img_normalized = img_resized / 255.0
        
        # Expand dimensions to create a batch of 1
        img_batch = np.expand_dims(img_normalized, axis=0)

        # --- 3. Predict ---
        print("Classifying image...")
        prediction_prob = model.predict(img_batch)[0][0]

        # --- 4. Display Result ---
        if prediction_prob > 0.5:
            label = "Fake"
            confidence = prediction_prob * 100
        else:
            label = "Real"
            confidence = (1 - prediction_prob) * 100

        print("\n--- Prediction Result ---")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Classification: {label}")
        print(f"Confidence: {confidence:.2f}%")
        print("-------------------------\n")

    except Exception as e:
        print(f"An error occurred during prediction: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify an image as Real or Fake.")
    parser.add_argument(
        "--image", 
        type=str, 
        required=True, 
        help="Path to the input image."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=os.path.join(os.path.dirname(__file__), '..', 'models', 'deepfake_detector_best.h5'),
        help="Path to the trained model file (.h5)."
    )
    
    args = parser.parse_args()
    
    predict_single_image(args.image, args.model)