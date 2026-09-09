from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ==========================================
# GRADIENT BOOSTING
# ==========================================

def train_gradient_boosting(X_train, y_train):
    """
    Train Gradient Boosting model.
    """

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate_gradient_boosting(model, X_test, y_test):
    """
    Evaluate Gradient Boosting model.
    """

    # Prediction
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
    print("GRADIENT BOOSTING")
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