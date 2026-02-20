## 3. Model Design & Architecture

### 3.1. CNN Architecture Choice

We will employ **transfer learning** by using a pre-trained CNN. This approach leverages a model that has already learned to recognize features from a massive dataset (like ImageNet) and fine-tunes it for our specific task.

*   **Chosen Architecture:** **ResNet-50** (Residual Network with 50 layers).
*   **Rationale:** ResNet-50 provides an excellent balance between performance and computational efficiency. Its residual connections help to mitigate the vanishing gradient problem in deep networks, allowing for effective training.

The ResNet-50 architecture will be modified for binary classification: the final fully connected layer will be replaced with a new one followed by a **Sigmoid activation function** to output a probability score between 0 (Real) and 1 (Deepfake).

### 3.2. Training Strategy and Hyperparameter Considerations

*   **Training Approach:** We will "unfreeze" the top layers of the pre-trained ResNet-50 and train them with a small learning rate, while keeping the earlier layers frozen. This fine-tuning approach adapts the high-level feature detectors of the network to our specific problem.
*   **Loss Function:** **Binary Cross-Entropy** will be used, as it is the standard for binary classification problems.
*   **Optimizer:** The **Adam optimizer** will be used for its efficiency and adaptive learning rate capabilities.
*   **Hyperparameters:**
    *   **Learning Rate:** A small initial learning rate (e.g., 1e-4) will be used for fine-tuning.
    *   **Batch Size:** A batch size of 32 or 64 will be used, depending on the available local GPU memory.
    *   **Epochs:** The model will be trained for a set number of epochs (e.g., 20-30), with **early stopping** monitored on the validation set's loss to prevent overfitting.
