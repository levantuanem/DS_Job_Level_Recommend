"""
Baseline Model - Logistic Regression

Người 4: Machine Learning Engineer

Mục đích:
- Xây dựng baseline bằng Logistic Regression
- Có thể nhận X_train, X_test, y_train, y_test từ bên ngoài

"""

import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def create_demo_data():
    """
    Tạo dữ liệu giả để test model

    """

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=12,
        n_redundant=3,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Chia dữ liệu thành train và test.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_baseline():
    """
    Tạo Baseline Logistic Regression.

    StandardScaler:
        Chuẩn hóa feature.

    LogisticRegression:
        Model baseline.
    """

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    return model


def train_baseline(X_train, y_train):
    """
    Train Logistic Regression.
    """

    model = build_baseline()

    model.fit(X_train, y_train)

    return model


def evaluate_baseline(model, X_test, y_test):
    """
    Đánh giá Baseline Model.
    """

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print("\n==============================")
    print("BASELINE - LOGISTIC REGRESSION")
    print("==============================")

    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision Macro: {precision:.4f}")
    print(f"Recall Macro   : {recall:.4f}")
    print(f"F1 Macro       : {f1_macro:.4f}")
    print(f"F1 Weighted    : {f1_weighted:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0,
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "model": "Logistic Regression",
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def main():
    """
    Chạy thử Baseline bằng dữ liệu giả.
    """

    print("Tạo dữ liệu demo...")

    X, y = create_demo_data()

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
    )

    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")

    model = train_baseline(
        X_train,
        y_train,
    )

    evaluate_baseline(
        model,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    main()