# AI-Based Image Deepfake Detection System

This project contains a complete, end-to-end solution for an AI-based image deepfake detection system. The model is built using TensorFlow and a ResNet50V2 architecture via transfer learning.

## Project Structure

- `docs/`: Contains all the project documentation, broken down into logical sections.
- `src/`: Contains the Python source code for the project.
- `data/`: Will be created to store raw and processed datasets.
- `models/`: Will be created to store the trained model and training history plots.
- `reports/`: Will be created to store evaluation reports and confusion matrices.
- `requirements.txt`: A list of all Python dependencies for the project.
- `README.md`: This file, providing an overview and operational instructions.

## How to Run the Project

Follow these steps in order to set up the environment, prepare the data, train the model, and run predictions.

### Step 0: Initial Setup

First, navigate to the project's root directory.

```bash
cd deepfake_detection_project
```

### Step 1: Set Up Python Environment

Create a virtual environment and install the required dependencies from `requirements.txt`.

```bash
# Create a Python virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate

# Install the required libraries
pip install -r requirements.txt
```
*Note: If you have an NVIDIA GPU, ensure you have CUDA and CuDNN installed for GPU acceleration with TensorFlow.*

### Step 2: Download and Extract Data

Run the `download_data.py` script. This will download the real (CelebA subset) and fake (TPDNE subset) image datasets into the `data/raw/` directory.

```bash
python src/download_data.py
```
*(This may take some time depending on your internet connection, as it downloads several hundred megabytes of data).*

### Step 3: Preprocess and Split Data

Run the `preprocess.py` script. This will resize the images and organize them into `train`, `validation`, and `test` sets inside the `data/processed/` directory.

```bash
python src/preprocess.py
```

### Step 4: Train the Model

Run the `train.py` script to begin training the neural network. The script will:
- Load the processed data.
- Build and compile the model.
- Train the model, saving the best version to `models/deepfake_detector_best.h5`.
- Save a plot of the training accuracy/loss to `models/training_history.png`.

```bash
python src/train.py
```
*(This is a computationally intensive step and will take a significant amount of time, especially without a GPU).*

### Step 5: Evaluate the Model

After training is complete, run the `evaluate.py` script to test the model's performance on the unseen test dataset.

```bash
python src/evaluate.py
```
This will print a classification report and save it, along with a confusion matrix plot, to the `reports/` directory.

### Step 6: Predict a Single Image

To classify your own image, use the `predict.py` script. Place your test image in the project's root directory or provide the full path to it.

```bash
# Example:
python src/predict.py --image /path/to/your/image.jpg
```

The script will load the trained model and output its prediction (Real or Fake) along with a confidence score.# ai-based-mage-authenticity-detector
