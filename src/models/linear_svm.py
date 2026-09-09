from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ==========================================
# LINEAR SVM
# ==========================================

def train_linear_svm(X_train, y_train):
    """
    Train Linear SVM model.
    """

    model = LinearSVC(
        C=1.0,
        random_state=42,
        max_iter=5000
    )

    model.fit(X_train, y_train)

    return model


def evaluate_linear_svm(model, X_test, y_test):
    """
    Evaluate Linear SVM model.
    """

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision_macro = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Print results
    print()
    print("=" * 40)
    print("LINEAR SVM")
    print("=" * 40)

    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision Macro: {precision_macro:.4f}")
    print(f"Recall Macro   : {recall_macro:.4f}")
    print(f"F1 Macro       : {f1_macro:.4f}")
    print(f"F1 Weighted    : {f1_weighted:.4f}")

    print()
    print("Classification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }