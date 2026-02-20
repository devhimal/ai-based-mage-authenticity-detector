## 4. Implementation Approach

### 4.1. Local Execution Workflow

The entire project will be implemented using Python and popular machine learning libraries.

1.  **Environment Setup:** A Python virtual environment will be created to manage dependencies. Key libraries will include:
    *   **TensorFlow** or **PyTorch** for model building and training.
    *   **NumPy** for numerical operations.
    *   **OpenCV** or **Pillow** for image processing.
    *   **Scikit-learn** for evaluation metrics.
    *   **Matplotlib** for plotting results.
2.  **Scripts:** The project will be organized into a series of Python scripts:
    *   `download_data.py`: Script to download and organize the datasets.
    *   `preprocess.py`: Script to perform the preprocessing steps and create the final dataset splits.
    *   `train.py`: The main script to build, train, and save the CNN model.
    *   `evaluate.py`: Script to load the trained model and evaluate it on the test set, generating a report.
    *   `predict.py`: A simple command-line interface to classify a single image.

### 4.2. User Interaction

Interaction will be script-based. The `predict.py` script will accept a path to an image file as a command-line argument and will output the classification (Real or Deepfake) and the confidence score.

`python predict.py --image /path/to/my_image.jpg`
