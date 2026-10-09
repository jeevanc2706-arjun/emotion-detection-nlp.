# Emotion Detection from Text

## Overview

An NLP-based text classification project for predicting emotion categories from textual input.

The project follows a standard NLP workflow using TF-IDF for feature extraction, SMOTE to address class imbalance, and Support Vector Machine (SVM) for classification.

> **Reconstructed portfolio implementation:** The original project files were no longer available. This repository is reconstructed from the project description in my resume and is intended to provide a reproducible implementation of the stated techniques.

## Technologies

- Python
- Pandas
- Scikit-learn
- TF-IDF
- SMOTE
- Support Vector Machine (SVM)
- Matplotlib
- Seaborn

## Workflow

Text Input → Cleaning → TF-IDF → Train/Test Split → SMOTE → SVM → Emotion Prediction → Evaluation

## Model

The implementation uses a linear SVM classifier with TF-IDF text features. SMOTE is applied to the training feature matrix to reduce the effect of class imbalance.

## Evaluation

The project reports:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Important Note

The original dataset and original project source files were not available. The included implementation uses a small reproducible sample dataset so the repository can be run and understood without claiming that these are the original project results.

## How to Run

```bash
pip install -r requirements.txt
python emotion_detection.py
```

## Project Structure

```text
emotion-detection-nlp/
├── README.md
├── emotion_detection.py
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── emotion_detection_analysis.ipynb
└── results/
    ├── confusion_matrix.png
    └── model_output.txt
```

## Author

Jeevan C
