## 6. Risk Mitigation

*   **Overfitting:** Mitigated through data augmentation, dropout layers in the classifier head, and early stopping based on validation loss.
*   **Dataset Bias:** The potential for demographic bias in the training data will be acknowledged and documented. The model's limitations in this regard will be clearly stated in the final report.
*   **False Positives/Negatives:** The evaluation will analyze the impact of both error types. For instance, in a disinformation context, a low recall (high false negatives) is highly undesirable. We will aim to tune the model to balance these errors according to the most critical use case.

## 7. Ethical & Societal Considerations

*   **Privacy:** We will only use datasets with licenses that permit academic research. No private or personally identifiable information will be collected or stored.
*   **Fairness:** We will document any known demographic biases in the dataset and discuss how they might affect the model's fairness. The model should not perform significantly worse for any particular group.
*   **Transparency:** The entire methodology, from dataset choice to model architecture and evaluation, will be documented to ensure reproducibility. We will not be using "black box" methods without justification.
*   **Responsible AI Use:** The final report will include a discussion on the potential for misuse of this technology (e.g., as a tool to lend credibility to "real" images in a disinformation campaign) and emphasize that this is a research prototype, not a production-ready system for making critical judgments.
