# 🩺 Automated Neural Network Pipeline for Diabetes Prediction

This repository contains a fully automated end-to-end PyTorch machine learning workflow designed to classify diabetes using the Pima Indians Diabetes dataset. The project strictly implements, regularizes, tunes, and contrasts two distinct neural network topologies: a Shallow Neural Network and a Deep Neural Network.

---

## 🚀 Key Features & Automation Standards
* **Zero Manual Intervention:** The pipeline executes seamlessly from start to finish via Google Colab's `Runtime -> Run all`.
* **Remote Data Ingestion:** Features absolute elimination of manual local file directory mappings. The source tabular data is dynamically pulled via its raw public repository URL.
* **Stratified Split & Preprocessing:** Handles impossible zero-value physiological entries via robust feature-specific median imputation and normalizes features using standardization ($Mean = 0, Std = 1$).

---

## 📊 Pipeline Architecture

### 1. Data Processing & EDA
* Continuous tracking of missing and zero-value configurations.
* Target class balance assessment via dynamic visualization matrix splits.
* Automated stratified train/test partitioning ($80/20$).

### 2. Model Topologies
* **Shallow Neural Network:** Contains exactly one hidden layer ($16$ units) utilizing a Rectified Linear Unit (ReLU) activation function, terminating at a standalone target prediction node mapping a Sigmoid layer.
* **Deep Neural Network:** Features four sequentially layered hidden blocks ($64 \rightarrow 32 \rightarrow 16 \rightarrow 8$ hidden nodes). Implements dropout regularization constraints ($0.3, 0.2$) alongside L2 weight decay ($1e-4$) to aggressively counteract dimensional overfitting.

---

## 📈 Visual Matrix & Subplots Included
The evaluation stage outputs the following evaluations in distinct formatted matrix subplots:
1. **Training History (2x1 Subplot Matrix):** Cross-comparison tracking of training versus validation losses and binary target accuracy curves over training iterations.
2. **Confusion Matrix (2x1 Heatmap Matrix):** Comparative seaborn heatmaps mapping True Positives, True Negatives, False Positives, and False Negatives.
3. **ROC Curve (2x1 Plot Matrix):** Receiver Operating Characteristic profiles explicitly displaying calculated Area Under the Curve (AUC) scores.
4. **Grouped Evaluation Performance Bar Chart:** Single unified layout contrasting performance metrics including Accuracy, Precision, Recall, F1-Score, and overall AUC.
5. **Sequential Topology Summary:** Text-based structural layout describing dimensions, dropout probability parameters, and tensor transforms.

---

## 🔬 Performance Highlights & Analytical Takeaway
* The execution histories reveal stable convergence across both model types. 
* Due to the concise, lower-dimensional layout of classic tabular medical metrics, the Shallow architecture reaches baseline effectiveness comparable to the Deep network, proving that excessively high structural depth is not mandatory for clean division boundaries on this dataset.
* Validation trends track training curves tightly, proving that the dropout layers and L2 regularizers properly isolated the deep model from memorizing artifacts or overfitting.

---

## 📄 Submission Direct Links
* **Dataset Source:** [Kaggle Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
* **GitHub Repository:** `https://github.com/just-abir/AI_ML_Lab_Task`
