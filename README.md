# Trustworthy AI for ECG Anomaly Detection through Adversarially Robust and Explainable Neural Network Models

---
OneDrive link (contains a copy of video, presentation and dataset_url) : 

https://livecoventryac-my.sharepoint.com/:f:/g/personal/binoyn_uni_coventry_ac_uk/IgCPx_tonYcpS4pX-S3RjT2GAW8jQFnY0Pou4oz4dzneQcc

## I. Project Overview

This project implements a complete deep learning pipeline for **ECG arrhythmia classification** using the MIT-BIH Arrhythmia dataset. The system processes raw ECG signals, segments individual heartbeats, and classifies them into clinically relevant categories defined by the AAMI standard.

Beyond standard classification, the project integrates:

* Deep learning architectures (CNN, ResNet, Transformer-CNN Hybrid)
* Advanced signal preprocessing
* Class imbalance handling using focal loss
* Explainable AI (Grad-CAM, Integrated Gradients)
* Adversarial attack simulation and defence (FGSM, PGD)

Unlike conventional approaches that focus solely on accuracy, this project adopts a **Trustworthy AI perspective**, evaluating models across:

* **Accuracy**
* **Robustness (adversarial stability)**
* **Interpretability (explanation reliability)**

---

## II. Dataset

* Dataset: MIT-BIH Arrhythmia Database
* Source: https://physionet.org/content/mitdb/1.0.0/

Heartbeat annotations are mapped into AAMI classes:

| Class | Description      |
| ----- | ---------------- |
| N     | Normal           |
| S     | Supraventricular |
| V     | Ventricular      |
| F     | Fusion           |
| Q     | Unknown          |

---

## III. Repository Structure

```
.
├── MIT_arrythmia.ipynb       # Complete pipeline implementation
├── README.md               # Documentation
├── dataset_url.txt         # Dataset source link
├── video.mp4               # Demonstration
├── ECG Arrhythmia Classification:
    A Trustworthy AI Approach.pptx  # Presentation
├── /models/                # Saved trained models and outputs
```

---

## IV. Technical Requirements & Setup

### Dependencies

```bash
pip install wfdb numpy pandas matplotlib pywavelets seaborn scikit-learn tensorflow
```

---

### Hardware Requirement (Recommended)

* Google Colab with **T4 GPU**

This is particularly important for:

* Transformer-CNN hybrid model
* Adversarial training (FGSM, PGD)

---

### Storage (Google Drive)

Model outputs are saved to Google Drive:

* Trained models
* Predictions
* Metrics (JSON)
* Confusion matrices

---

## V. Implementation Summary

### 1. Signal Preprocessing

Each ECG signal undergoes:

* Baseline Wander Removal
* Powerline Noise Removal (60Hz Notch)
* Bandpass Filtering (0.5–40 Hz)
* Wavelet Denoising (Sym4)
* Z-score Normalisation

---

### 2. Beat Segmentation

* R-peak detection from annotations
* Window size: ±180 samples
* Final input: **360-length ECG segment**

---

### 3. Model Architectures

#### CNN Models

* Base CNN
* Hypertuned CNN (improved stability and minority class handling)

#### ResNet Models (1D)

* Base ResNet
* Hypertuned ResNet (deeper feature representation)

#### Transformer-CNN Hybrid

* Combines convolutional feature extraction with attention mechanisms
* Captures both local and global ECG dependencies

---

### 4. Class Imbalance Handling

* Class weights computed using `sklearn`
* Custom **Focal Loss**:

  * Alpha: class balancing
  * Gamma: focus on hard samples
* Threshold tuning for critical classes

---

## VI. Explainable AI (XAI)

### Grad-CAM

* Highlights important ECG regions influencing predictions

### Integrated Gradients (IG)

* Quantifies feature attribution across time-series signals

 **Key Insight:**
Models that exhibit higher robustness also produce **more stable and clinically meaningful explanations**, demonstrating that interpretability is closely linked to robustness.

---

## VII. Adversarial Robustness & Defence

### Adversarial Attacks

* FGSM (single-step perturbation)
* PGD (iterative perturbation)

### Defence Strategy

* Adversarial training using perturbed ECG signals
* Improves robustness and reduces performance degradation

 **Key Insight:**
Adversarial perturbations affect both predictions **and model reasoning**, highlighting the importance of evaluating explainability under adversarial conditions.

---

## VIII. Training Pipeline

```python
train_and_save(model, model_name, ...)
```

### Features:

* Early stopping
* Learning rate scheduling
* Model checkpointing
* Automatic saving to Google Drive

---

## IX. Usage Instructions

### Step 1

Run `MIT_arrythmia.ipynb`

### Step 2

Enable **T4 GPU**

### Step 3

Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4

Run all cells

---

## X. Results Summary

### Metrics Used for model accuracy evaluation

* Accuracy
* Macro F1 Score (primary metric due to class imbalance)
* Normalised Confusion Matrix

---

### Key Findings (Critical Insights)

* **CNN models provide the most stable and robust performance**, with minimal degradation under adversarial conditions
* **ResNet improves feature representation but increases sensitivity to perturbations**
* **Transformer-CNN Hybrid shows highest performance degradation**, indicating that increased complexity does not guarantee robustness

 **Core Finding:**
There exists a fundamental trade-off between **accuracy, robustness, and interpretability** across architectures.

---

### Adversarial Robustness Insights

* CNN: minimal degradation (most stable)
* ResNet: moderate degradation
* Hybrid: highest degradation under FGSM/PGD

---

### Explainability AI Insights

* Stable models (CNN) show consistent attention regions (QRS complex)
* Less robust models exhibit **significant explanation shifts**
* Explanation stability is directly influenced by adversarial robustness

---

## XI. Reproducibility

* Deterministic train-test split
* Saved datasets and predictions
* Modular training pipeline
* Stored results for verification

---

## XII. Limitations

* Evaluation limited to a **single dataset (MIT-BIH)**
* Adversarial defence restricted to **FGSM-based and PGD-based training**
* Hybrid model not fully optimised relative to complexity
* Explanation similarity metrics occasionally produce instability (NaN cases)
* Computational overhead due to adversarial training and hybrid architectures

---

## XIII. Future Work

* Implement stronger defence strategies (e.g., TRADES)
* Extend evaluation across multiple ECG datasets
* Improve hybrid model optimisation
* Explore causal and model-intrinsic explainability methods
* Develop real-time clinical deployment systems

---

## XIV. Contribution

This project makes the following contributions:

* Proposes a **unified Trustworthy AI framework** integrating accuracy, robustness, and interpretability
* Introduces **quantitative evaluation of explanation stability**
* Demonstrates that **robustness is a prerequisite for reliable explainability**
* Provides new insight that **increased model complexity does not guarantee trustworthy performance**
* Links model explanations to **clinically relevant ECG features**

---

## XV. Notes

* This README documents the **artefact (implementation and usage)**
* Academic justification is provided in the accompanying dissertation report

---
