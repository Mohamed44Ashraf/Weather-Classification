# 🌧️ RainTomorrow Prediction Project

## 📌 Overview

This project predicts whether it will rain **tomorrow** (`RainTomorrow`) using historical weather data.
The pipeline includes:

* EDA for raw Data
* Impute Missing ValUes
* Handle Outlier
* Feature engineering
* Visualization and Analysis
* Encoding
* Data preprocessing (scaling, handling imbalance with SMOTE)
* Model training with **CatBoost Classifier**
* Hyperparameter tuning with **GridSearchCV**
* Model evaluation using cross-validation (Accuracy, Precision, Recall, F1-score)
* Performance reports (Confusion Matrix & Classification Report)

---

## ⚙️ Requirements

Install the required libraries before running the project:

```bash
pip install pandas numpy scikit-learn catboost imbalanced-learn matplotlib seaborn
```

---

## 📂 Project Structure

```
Weather_classification/
 ├── dataset/
 │     ├── raw/
 ├── src/
 │     ├── data_preprocessing.py
 │     ├── train.py
 ├── workspace/
 │     ├── notebooks/
 │     │     └── experiments.ipynb
 │     ├── models/
 │     │     ├── catboost_model.pkl
 │     │     ├── scaler.pkl
 │     │     └── encoder.pkl
 │     └── figures/
 └── README.md
 └── requirements.txt       # List of dependencies
---

## 🚀 How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/Weather_Classification.git
   cd Weather_Classification
   ```

2. Place your dataset inside the `data/` folder.

3. Run the training script:

   ```bash
   python src/train.py
   ```

---

## 📊 Model Evaluation

Metrics calculated:

* Accuracy
* Precision
* Recall
* F1-score

Additionally, the project generates:

* ✅ **Confusion Matrix**
* ✅ **Classification Report**

Example output:

```
Accuracy: 0.85
Precision: 0.73
Recall: 0.57
F1-score: 0.64
```

Confusion Matrix example:

|                | Predicted No | Predicted Yes |
| -------------- | ------------ | ------------- |
| **Actual No**  | 25941        | 1629          |
| **Actual Yes** | 3427         | 4551          |

---

## 📌 Next Steps

* Try feature selection techniques to improve performance.
* Deploy the model as a simple **Flask API** or interactive **Streamlit dashboard**.

---

## 👨‍💻 Author

* **Mohamed Ashraf Ezzat**
  Data Scientist & Coding instructor | Faculty of Computers & AI, Benha University
