## 8. Deliverables Mapping

| Deliverable             | Corresponding Implementation Artifact                                   |
| ----------------------- | --------------------------------------------------------------------- |
| **Dataset Module**      | `download_data.py` and `preprocess.py` scripts and the resulting data directories. |
| **AI Model**            | The saved, trained model file (e.g., `deepfake_detector.h5`).         |
| **Evaluation Report**   | A Jupyter Notebook or PDF (`evaluation_report.pdf`) containing metrics, charts, and analysis. |
| **User Interaction**    | `predict.py` script for single image classification.                  |
| **Final Documentation** | This document (`AI_Deepfake_Detection_Solution.md`).                  |
| **Presentation**        | A presentation summarizing the project, derived from this document.     |

## 9. Milestone-Based Execution Plan

This project is planned over a 4-week timeline.

| Milestone | Week | Key Tasks                                                                  |
| :-------- | :--- | :------------------------------------------------------------------------- |
| **1: Foundation & Data** | 1    | - Finalize project scope and set up the development environment. <br> - Research, select, and download datasets. <br> - Implement and run `download_data.py` and `preprocess.py`. |
| **2: Model Development** | 2    | - Implement the ResNet-50 based model architecture. <br> - Write the initial `train.py` script. <br> - Begin initial training runs to debug the pipeline. |
| **3: Training & Evaluation** | 3  | - Conduct full training runs with hyperparameter tuning on the validation set. <br> - Implement the `evaluate.py` script. <br> - Run evaluation on the test set and generate the `evaluation_report`. |
| **4: Finalization** | 4    | - Analyze results and document findings. <br> - Implement the `predict.py` user interaction script. <br> - Finalize all documentation and prepare the presentation. |
