# src/evaluate.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(img_size=(256, 256), batch_size=32):
    """
    Evaluates the trained model on the unseen test set.
    - Loads the best saved model.
    - Makes predictions on the test data.
    - Prints a classification report and displays a confusion matrix.
    """
    print("Starting model evaluation on the test set...")

    base_dir = os.path.join(os.path.dirname(__file__), '..')
    processed_data_dir = os.path.join(base_dir, 'data', 'processed')
    model_path = os.path.join(base_dir, 'models', 'deepfake_detector_best.h5')
    report_save_dir = os.path.join(base_dir, 'reports')
    os.makedirs(report_save_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please train the model first.")
        return

    # --- 1. Load Model ---
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # --- 2. Load Test Data ---
    # Note: No data augmentation is applied to the test set, only rescaling.
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        os.path.join(processed_data_dir, 'test'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False  # Important: Do not shuffle test data
    )

    # --- 3. Generate Predictions ---
    print("Generating predictions on the test set...")
    # Get the ground truth labels
    y_true = test_generator.classes
    
    # Predict probabilities
    y_pred_probs = model.predict(test_generator, steps=np.ceil(test_generator.samples / batch_size))
    
    # Convert probabilities to class labels (0 or 1)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # --- 4. Generate Classification Report and Confusion Matrix ---
    class_labels = list(test_generator.class_indices.keys())
    
    # Classification Report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print(report)
    
    # Save report to a file
    report_path = os.path.join(report_save_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Plot and save the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    
    cm_plot_path = os.path.join(report_save_dir, 'confusion_matrix.png')
    plt.savefig(cm_plot_path)
    print(f"Confusion matrix plot saved to {cm_plot_path}")
    
    print("\nModel evaluation complete.")

if __name__ == "__main__":
    evaluate_model()