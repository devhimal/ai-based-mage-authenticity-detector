import os
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import cv2
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Basic configuration
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Model Loading ---
# We will load the model here later.
model = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'deepfake_detector_best.h5')
IMG_SIZE = (96, 96)

def load_model():
    """Load the trained model."""
    global model
    print(" * Loading model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(" * Model loaded successfully.")
    except Exception as e:
        print(f" * Error loading model: {e}")
        # Handle case where model doesn't exist yet
        model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # --- Prediction Logic ---
            if model is None:
                # Handle case where model is not loaded
                return render_template('index.html', error="Model is not loaded. Please check the server logs.")

            # Read and preprocess the image
            img = cv2.imread(filepath)
            img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
            img_normalized = img_resized / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)

            # Predict
            prediction_prob = model.predict(img_batch)[0][0]

            # Format the result
            if prediction_prob > 0.5:
                label = "Fake"
                confidence = prediction_prob * 100
            else:
                label = "Real"
                confidence = (1 - prediction_prob) * 100
            
            prediction_result = {
                'label': label,
                'confidence': f"{confidence:.2f}"
            }
            
            # Render the page with the prediction result
            return render_template('index.html', prediction=prediction_result, img_path=filepath)

    return render_template('index.html')


if __name__ == '__main__':
    load_model()  # Load the model when the app starts
    app.run(debug=True)
