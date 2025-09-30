
import os
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from preprocess import load_data, preprocessing


def encode_categorical(df):
    cat_cols = df.select_dtypes(include="object").columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    df[cat_cols] = df[cat_cols].astype("int32")
    return df, encoders


def split_features(df):
    X = df.drop(columns=["RainTomorrow"])
    y = df["RainTomorrow"]
    return X, y


def train_model(X, y):
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    smote = SMOTE(random_state=42)
    catboost = CatBoostClassifier(verbose=0, task_type="CPU")

    acc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []

    last_y_test, last_preds = None, None

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Balance classes
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train
        catboost.fit(X_train, y_train)
        preds = catboost.predict(X_test)

        # Metrics
        acc_scores.append(accuracy_score(y_test, preds))
        prec_scores.append(precision_score(y_test, preds, zero_division=0))
        rec_scores.append(recall_score(y_test, preds, zero_division=0))
        f1_scores.append(f1_score(y_test, preds, zero_division=0))

        last_y_test, last_preds = y_test, preds

    acc_avg = np.mean(acc_scores)
    prec_avg = np.mean(prec_scores)
    rec_avg = np.mean(rec_scores)
    f1_avg = np.mean(f1_scores)

    print("Final Results with CatBoost + SMOTE + KFold")
    print("=" * 50)
    print(f"Accuracy : {acc_avg:.4f}")
    print(f"Precision: {prec_avg:.4f}")
    print(f"Recall   : {rec_avg:.4f}")
    print(f"F1 Score : {f1_avg:.4f}")

    return catboost, scaler, (acc_avg, prec_avg, rec_avg, f1_avg), last_y_test, last_preds


def show_metrics(y_test, preds):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    cr = classification_report(y_test, preds, zero_division=0)
    print("Classification Report:\n", cr)


if __name__ == "__main__":
    df = load_data("D:\Track projects\Weather_Classification\dataset\weatherAUS.csv")

    df = preprocessing(df)
    
    df, encoders = encode_categorical(df)
    X, y = split_features(df)

    model, scaler, metrics, y_test, preds = train_model(X, y)

    show_metrics(y_test, preds)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/catboost_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(encoders, "models/encoders.pkl")
    print("Models saved successfully in 'models/' folder.")

