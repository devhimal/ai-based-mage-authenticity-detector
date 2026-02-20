## 5. Evaluation Methodology

### 5.1. Performance Metrics

The model's performance on the unseen test set will be evaluated using a standard set of classification metrics:

*   **Accuracy:** The overall percentage of correctly classified images.
*   **Precision:** Of all images predicted as Deepfake, what proportion were actually Deepfakes.
*   **Recall (Sensitivity):** Of all actual Deepfake images, what proportion were correctly identified.
*   **F1-Score:** The harmonic mean of Precision and Recall, providing a single score that balances both.
*   **Confusion Matrix:** A table showing the number of True Positives, True Negatives, False Positives, and False Negatives, giving a detailed view of classification performance.

### 5.2. Interpretation of Results

The results will be compiled into an evaluation report. We will pay close attention to the trade-off between precision and recall, as false negatives (classifying a deepfake as real) and false positives (classifying a real image as a deepfake) have different implications. The goal is to achieve a high F1-score, indicating a robust and balanced model.
